import asyncio
import csv
import gc
import joblib
import json
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama

# ============================================================
# CONFIG
# ============================================================

# orchestrator.pkl + label_mapping.json
ORCHESTRATOR_MODEL_PATH = "./models/"

# Map each label to its GGUF model path
GGUF_MODELS = {
    "anxiety": "./models/gguf/anxiety-q8_0.gguf",
    "bipolar": "./models/gguf/bipolar-q8_0.gguf",
    "depression": "./models/gguf/depression-q8_0.gguf",
    "ocd": "./models/gguf/ocd-q8_0.gguf",
    "schizophrenia": "./models/gguf/schizophrenia-q8_0.gguf",
}

# Hard limit on how many GGUF models can be resident at once
ENABLE_MEMORY_LIMIT = True
MAX_LOADED_MODELS = 5

# Batching per expert
ENABLE_BATCHING = True
MAX_BATCH_SIZE = 4
MAX_BATCH_WAIT_S = 0.1

# llama.cpp params
N_CTX = 4096
N_THREADS = 8
N_PARALLEL = 1

LOG_FILE_SERVER = "./logs/server_gguf_logs_azure.csv"


# ============================================================
# Logging infra
# ============================================================

class CsvLoggerThread(threading.Thread):
    def __init__(self, log_queue: queue.Queue, filename: str, fieldnames: List[str]):
        super().__init__(daemon=True)
        self.log_queue = log_queue
        self.filename = filename
        self.fieldnames = fieldnames

    def run(self):
        os.makedirs(os.path.dirname(self.filename) or ".", exist_ok=True)
        file_exists = os.path.exists(self.filename)
        with open(self.filename, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if not file_exists:
                writer.writeheader()
                f.flush()
            while True:
                record = self.log_queue.get()
                if record is None:
                    break
                writer.writerow(record)
                f.flush()


server_log_queue = queue.Queue()
server_logger = CsvLoggerThread(
    server_log_queue,
    LOG_FILE_SERVER,
    fieldnames=[
        "server_receive_ts",
        "request_id",
        "func_id",
        "prompt_len",
        "chosen_label",
        "chosen_expert",
        "orchestrator_time_ms",
        "preprocess_time_ms",
        "queue_time_ms",
        "load_time_ms",
        "inference_time_ms",
        "total_time_ms",
        "batch_size",
    ],
)
server_logger.start()


def log_server(record: Dict):
    server_log_queue.put(record)


# ============================================================
# Orchestrator
# ============================================================

class OrchestratorRouter:
    def __init__(self, model_dir: str):
        labelmap_path = os.path.join(model_dir, "label_mapping.json")
        model_path = os.path.join(model_dir, "orchestrator.pkl")

        pipe = joblib.load(model_path)
        with open(labelmap_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)

        self.model = pipe
        self.label2id = mapping["LABEL2ID"]
        self.id2label = {int(k): v for k, v in mapping["ID2LABEL"].items()}

    def route(self, text: str) -> str:
        pred = self.model.predict([text])
        label_id = pred[0]
        return self.id2label[label_id]


# ============================================================
# Global model state for memory limiting
# ============================================================

experts: Dict[str, "GgufExpert"] = {}  # filled on startup

# Protects: num_loaded, expert.active_count, expert.llm
global_cond = threading.Condition()
num_loaded = 0


# ============================================================
# Expert models (GGUF via llama.cpp)
# ============================================================

@dataclass
class BatchJob:
    request_id: str
    func_id: int
    prompt: str
    enqueue_time: float
    future: asyncio.Future


@dataclass
class GgufExpert:
    name: str
    model_path: str

    llm: Optional[Llama] = field(default=None, init=False)
    active_count: int = field(default=0, init=False)
    
    request_queue: Optional[asyncio.Queue] = field(default=None, init=False)
    worker_task: Optional[asyncio.Task] = field(default=None, init=False)

    # ---------- load accounting ----------

    def inc_active(self):
        global global_cond
        with global_cond:
            self.active_count += 1

    def dec_active(self):
        global global_cond
        with global_cond:
            self.active_count = max(0, self.active_count - 1)
            global_cond.notify_all()

    # ---------- memory-limited loading ----------

    def ensure_loaded(self) -> float:
        """
        Hard memory limit:
        - At most MAX_LOADED_MODELS experts with llm != None.
        - If limit is hit:
            * Try to evict some other idle expert (active_count == 0).
            * If none idle, WAIT until someone finishes and retry.
        """
        global num_loaded, global_cond, experts, ENABLE_MEMORY_LIMIT, MAX_LOADED_MODELS

        start = time.perf_counter()

        with global_cond:
            if self.llm is not None:
                return 0.0

            while True:
                if self.llm is not None:
                    return 0.0

                if not ENABLE_MEMORY_LIMIT:
                    num_loaded += 1
                    break

                if num_loaded < MAX_LOADED_MODELS:
                    num_loaded += 1
                    break

                # Limit reached: try to evict some other idle expert
                idle_exp: Optional[GgufExpert] = None
                for other in experts.values():
                    if other is self:
                        continue
                    if other.llm is not None and other.active_count == 0:
                        idle_exp = other
                        break

                if idle_exp is not None:
                    print(f"[MEM] Evicting idle expert '{idle_exp.name}'")
                    idle_exp.llm = None
                    num_loaded = max(0, num_loaded - 1)
                    gc.collect()
                    continue

                # All loaded experts busy: wait
                global_cond.wait()

        # We reserved a slot (num_loaded++). Load outside lock.
        try:
            print(f"[MEM] Loading GGUF for expert '{self.name}' from {self.model_path}")
            llm = Llama(
                model_path=self.model_path,
                n_ctx=N_CTX,
                n_threads=N_THREADS,
                n_batch=MAX_BATCH_SIZE * 32,
                n_gpu_layers=0,
                logits_all=False,
                vocab_only=False,
                seed=0,
            )
        except Exception:
            with global_cond:
                num_loaded = max(0, num_loaded - 1)
                global_cond.notify_all()
            raise

        with global_cond:
            self.llm = llm
            global_cond.notify_all()

        end = time.perf_counter()
        return (end - start) * 1000.0

    # ---------- unload ----------

    def unload(self):
        global num_loaded, global_cond
        with global_cond:
            if self.llm is not None:
                print(f"[MEM] Unloading expert '{self.name}'")
                self.llm = None
                num_loaded = max(0, num_loaded - 1)
                gc.collect()
                global_cond.notify_all()

    # ---------- inference ----------

    def _generate_one(self, prompt: str, max_new_tokens: int = 128) -> str:
        assert self.llm is not None, "llm not loaded"
        result = self.llm.create_chat_completion(
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_new_tokens,
            temperature=0.8,
            top_p=0.9,
        )
        return result["choices"][0]["message"]["content"].strip()

    def generate_single(self, prompt: str, max_new_tokens: int = 128) -> str:
        return self._generate_one(prompt, max_new_tokens=max_new_tokens)

    def generate_batch_sequential(self, prompts: List[str], max_new_tokens: int = 128) -> List[str]:
        outputs = []
        for p in prompts:
            outputs.append(self._generate_one(p, max_new_tokens=max_new_tokens))
        return outputs


# ============================================================
# FastAPI schemas
# ============================================================

class InferenceRequest(BaseModel):
    request_id: str
    func_id: int
    start_time: float
    prompt: str


class InferenceResponse(BaseModel):
    request_id: str
    func_id: int
    prompt: str
    output: str
    chosen_label: str
    chosen_expert: str
    timings_ms: Dict[str, float]
    batch_size: int


# ============================================================
# App setup
# ============================================================

app = FastAPI()

orchestrator: OrchestratorRouter


@app.on_event("startup")
async def startup_event():
    global orchestrator, experts, num_loaded

    orchestrator = OrchestratorRouter(ORCHESTRATOR_MODEL_PATH)

    experts = {
        label: GgufExpert(
            name=label,
            model_path=gguf_path,
        )
        for label, gguf_path in GGUF_MODELS.items()
    }

    num_loaded = 0

    if ENABLE_BATCHING:
        loop = asyncio.get_event_loop()
        for exp in experts.values():
            exp.request_queue = asyncio.Queue()
            exp.worker_task = loop.create_task(batch_worker(exp))


def elapsed_ms(a: float, b: float) -> float:
    return (b - a) * 1000.0


# ============================================================
# Batching worker (per expert)
# ============================================================

async def batch_worker(exp: GgufExpert):
    assert exp.request_queue is not None
    q = exp.request_queue

    while True:
        job: BatchJob = await q.get()
        batch: List[BatchJob] = [job]

        batch_collect_start = time.perf_counter()
        while len(batch) < MAX_BATCH_SIZE:
            remaining = MAX_BATCH_WAIT_S - (time.perf_counter() - batch_collect_start)
            if remaining <= 0:
                break
            try:
                nxt = await asyncio.wait_for(q.get(), timeout=remaining)
                batch.append(nxt)
            except asyncio.TimeoutError:
                break

        batch_size = len(batch)
        batch_start_time = time.perf_counter()

        exp.inc_active()
        try:
            load_time_ms = await asyncio.to_thread(exp.ensure_loaded)

            inf_start = time.perf_counter()
            prompts = [b.prompt for b in batch]

            outputs = await asyncio.to_thread(exp.generate_batch_sequential, prompts)
            inf_end = time.perf_counter()
        finally:
            exp.dec_active()

        for b, out in zip(batch, outputs):
            queue_time_ms = elapsed_ms(b.enqueue_time, batch_start_time)
            inf_time_ms = elapsed_ms(inf_start, inf_end)
            if not b.future.done():
                b.future.set_result(
                    (out, load_time_ms, queue_time_ms, inf_time_ms, batch_size)
                )
        q.task_done()


# ============================================================
# Main endpoint
# ============================================================

@app.post("/infer", response_model=InferenceResponse)
async def infer(req: InferenceRequest):
    t0 = time.perf_counter()
    preprocess_start = t0

    orch_start = time.perf_counter()
    label = orchestrator.route(req.prompt)
    orch_end = time.perf_counter()
    preprocess_end = orch_end

    chosen_expert = experts[label]

    if ENABLE_BATCHING:
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        enqueue_time = time.perf_counter()
        job = BatchJob(
            request_id=req.request_id,
            func_id=req.func_id,
            prompt=req.prompt,
            enqueue_time=enqueue_time,
            future=fut,
        )
        await chosen_expert.request_queue.put(job)

        output, load_time_ms, queue_time_ms, inf_time_ms, batch_size = await fut
    else:
        queue_time_ms = 0.0
        batch_size = 1

        chosen_expert.inc_active()
        try:
            load_time_ms = await asyncio.to_thread(chosen_expert.ensure_loaded)

            inf_start = time.perf_counter()
            output = await asyncio.to_thread(chosen_expert.generate_single, req.prompt)
            inf_end = time.perf_counter()
        finally:
            chosen_expert.dec_active()

        inf_time_ms = elapsed_ms(inf_start, inf_end)

    t_end = time.perf_counter()

    timings = {
        "orchestrator_time_ms": elapsed_ms(orch_start, orch_end),
        "preprocess_time_ms": elapsed_ms(preprocess_start, preprocess_end),
        "queue_time_ms": queue_time_ms,
        "load_time_ms": load_time_ms,
        "inference_time_ms": inf_time_ms,
        "total_time_ms": elapsed_ms(t0, t_end),
    }

    log_server(
        {
            "server_receive_ts": time.time(),
            "request_id": req.request_id,
            "func_id": req.func_id,
            "prompt_len": len(req.prompt),
            "chosen_label": label,
            "chosen_expert": chosen_expert.name,
            "batch_size": batch_size,
            **timings,
        }
    )

    return InferenceResponse(
        request_id=req.request_id,
        func_id=req.func_id,
        prompt=req.prompt,
        output=output,
        chosen_label=label,
        chosen_expert=chosen_expert.name,
        timings_ms=timings,
        batch_size=batch_size,
    )

# run with:
#   python -m uvicorn server_gguf:app --host 0.0.0.0 --port 8000
