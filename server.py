import asyncio
import csv
import gc
import joblib
import json
import os
import queue
import threading
import time
import torch

from dataclasses import dataclass, field
from fastapi import FastAPI
from peft import PeftModel
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from typing import Dict, List, Optional

# ============================================================
# CONFIG
# ============================================================

MODELS_DIR = "./models/"

BASE_LM_NAME = "meta-llama/Llama-3.2-1B-Instruct"
BASE_MODEL_PATH = "./models/Llama-3.2-1B-Instruct"

# orchestrator.pkl + label_mapping.json
ORCHESTRATOR_MODEL_PATH = MODELS_DIR

EXPERT_ADAPTERS = {
    "anxiety": os.path.join(MODELS_DIR, "anxiety"),
    "bipolar": os.path.join(MODELS_DIR, "bipolar"),
    "depression": os.path.join(MODELS_DIR, "depression"),
    "ocd": os.path.join(MODELS_DIR, "ocd"),
    "schizophrenia": os.path.join(MODELS_DIR, "schizophrenia"),
}

LABEL2ID = {
    "anxiety": 0,
    "bipolar": 1,
    "depression": 2,
    "ocd": 3,
    "schizophrenia": 4,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

DEVICE = torch.device("cpu")  # CPU-only

# --- toggles for experiments ---

ENABLE_MEMORY_LIMIT = True
MAX_LOADED_EXPERTS = 2  # strict hard limit

ENABLE_BATCHING = True
MAX_BATCH_SIZE = 4
MAX_BATCH_WAIT_S = 0.1

LOG_FILE_SERVER = "./logs/server_logs_testing.csv"

HF_TOKEN = "" 

# ============================================================
# Logging infra
# ============================================================

class CsvLoggerThread(threading.Thread):
    def __init__(self, log_queue: queue.Queue, filename: str, fieldnames):
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


def log_server(record):
    server_log_queue.put(record)


# ============================================================
# Orchestrator
# ============================================================

class OrchestratorRouter:
    def __init__(self, model_path: str):
        labelmap_path = os.path.join(model_path, "label_mapping.json")
        model_path = os.path.join(model_path, "orchestrator.pkl")

        pipe = joblib.load(model_path)
        with open(labelmap_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        label2id = mapping["LABEL2ID"]
        id2label = {int(k): v for k, v in mapping["ID2LABEL"].items()}

        self.model = pipe
        self.label2id = label2id
        self.id2label = id2label

    def route(self, text: str) -> str:
        probs = self.model.predict([text])
        label_id = probs[0]
        label_str = self.id2label[label_id]
        return label_str


# ============================================================
# Global model state for memory limiting
# ============================================================

experts: Dict[str, "ExpertModel"] = {}
global_cond = threading.Condition()
num_loaded = 0


# ============================================================
# Expert models
# ============================================================

@dataclass
class BatchJob:
    request_id: str
    func_id: int
    prompt: str
    enqueue_time: float
    future: asyncio.Future


@dataclass
class ExpertModel:
    name: str
    adapter_path: str
    tokenizer: AutoTokenizer
    base_lm_name: str = BASE_LM_NAME

    model: Optional[AutoModelForCausalLM] = field(default=None, init=False)
    active_count: int = field(default=0, init=False)

    request_queue: Optional[asyncio.Queue] = field(default=None, init=False)
    worker_task: Optional[asyncio.Task] = field(default=None, init=False)

    # ------------ usage tracking ------------
    def acquire_use(self):
        global global_cond
        with global_cond:
            self.active_count += 1

    def release_use(self):
        global global_cond
        with global_cond:
            self.active_count = max(0, self.active_count - 1)
            global_cond.notify_all()

    # ------------ memory-limited loading ------------

    def ensure_loaded(self) -> float:
        """
        Hard memory limit:
        - At most MAX_LOADED_EXPERTS experts with model != None.
        - If limit is hit:
            * Try to evict some other idle expert (active_count == 0).
            * If none idle, WAIT until someone finishes (release_use) and retry.
        """
        global num_loaded, global_cond, experts, ENABLE_MEMORY_LIMIT, MAX_LOADED_EXPERTS

        start = time.perf_counter()

        with global_cond:
            if self.model is not None:
                return 0.0

            while True:
                if self.model is not None:
                    return 0.0

                if not ENABLE_MEMORY_LIMIT:
                    num_loaded += 1
                    break

                if num_loaded < MAX_LOADED_EXPERTS:
                    num_loaded += 1
                    break

                idle_exp: Optional[ExpertModel] = None
                for other in experts.values():
                    if other is self:
                        continue
                    if other.model is not None and other.active_count == 0:
                        idle_exp = other
                        break

                if idle_exp is not None:
                    print(f"[MEM] Evicting idle expert '{idle_exp.name}'")
                    idle_exp.model = None
                    num_loaded = max(0, num_loaded - 1)
                    gc.collect()
                    continue

                global_cond.wait()

        try:
            print(f"[MEM] Loading expert '{self.name}' from {self.adapter_path}...")
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_PATH,
                torch_dtype=torch.float32,
                device_map={"": DEVICE},
                token=HF_TOKEN,
            )
            print(f"[MEM] Base model loaded for expert '{self.name}'")
            model = PeftModel.from_pretrained(
                base_model,
                self.adapter_path,
            ).to(DEVICE)
            model.eval()
        except Exception:
            with global_cond:
                num_loaded = max(0, num_loaded - 1)
                global_cond.notify_all()
            raise

        with global_cond:
            self.model = model
            global_cond.notify_all()

        end = time.perf_counter()
        return (end - start) * 1000.0

    def unload(self):
        global num_loaded, global_cond
        with global_cond:
            if self.model is not None:
                print(f"[MEM] Unloading expert '{self.name}'")
                self.model = None
                num_loaded = max(0, num_loaded - 1)
                gc.collect()
                global_cond.notify_all()

    # ------------ generation helpers ------------

    def generate_batch(self, prompts: List[str], max_new_tokens: int = 128) -> List[str]:
        assert self.model is not None, "Model not loaded"

        conversations = [[{"role": "user", "content": p}] for p in prompts]

        input_ids = self.tokenizer.apply_chat_template(
            conversations,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_generation_prompt=True,
        ).to(DEVICE)

        # Explicit attention mask because pad_token == eos_token
        attn_mask = (input_ids != self.tokenizer.pad_token_id).long().to(DEVICE)
        input_len = input_ids.shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        gen_only = output_ids[:, input_len:]
        texts = self.tokenizer.batch_decode(gen_only, skip_special_tokens=True)
        return [t.strip() for t in texts]

    def generate_single(self, prompt: str, max_new_tokens: int = 128) -> str:
        return self.generate_batch([prompt], max_new_tokens=max_new_tokens)[0]


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

    base_tokenizer = AutoTokenizer.from_pretrained(BASE_LM_NAME)
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
    base_tokenizer.padding_side = "left"

    experts = {
        label: ExpertModel(
            name=label,
            adapter_path=adapter_path,
            tokenizer=base_tokenizer,
        )
        for label, adapter_path in EXPERT_ADAPTERS.items()
    }

    num_loaded = 0

    if ENABLE_BATCHING:
        loop = asyncio.get_event_loop()
        for exp in experts.values():
            exp.request_queue = asyncio.Queue()
            exp.worker_task = loop.create_task(batch_worker(exp))


def elapsed_ms(start: float, end: float) -> float:
    return (end - start) * 1000.0


# ============================================================
# Batching worker (if you enable batching)
# ============================================================

async def batch_worker(exp: ExpertModel):
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
                next_job = await asyncio.wait_for(q.get(), timeout=remaining)
                batch.append(next_job)
            except asyncio.TimeoutError:
                break

        batch_size = len(batch)
        batch_start_time = time.perf_counter()

        exp.acquire_use()
        try:
            load_time_ms = await asyncio.to_thread(exp.ensure_loaded)

            inf_start = time.perf_counter()
            prompts = [b.prompt for b in batch]
            outputs = await asyncio.to_thread(exp.generate_batch, prompts)
            inf_end = time.perf_counter()
        finally:
            exp.release_use()

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

    # --- orchestrator ---
    orch_start = time.perf_counter()
    label = orchestrator.route(req.prompt)
    orch_end = time.perf_counter()

    chosen_expert = experts[label]
    preprocess_end = orch_end

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

        exp = chosen_expert
        exp.acquire_use()
        try:
            load_time_ms = await asyncio.to_thread(exp.ensure_loaded)

            inf_start = time.perf_counter()
            output = await asyncio.to_thread(exp.generate_single, req.prompt)
            inf_end = time.perf_counter()
        finally:
            exp.release_use()

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
#   python -m uvicorn server:app --host 0.0.0.0 --port 8000
