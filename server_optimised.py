import asyncio
import csv
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import joblib
import json

# =============================
# CONFIG
# =============================

MODELS_DIR = "./models/"

BASE_LM_NAME = "meta-llama/Llama-3.2-1B-Instruct"
BASE_MODEL_PATH = "./models/Llama-3.2-1B-Instruct"

ORCHESTRATOR_MODEL_PATH = MODELS_DIR  # orchestrator.pkl + label_mapping.json

EXPERT_ADAPTERS = {
    "anxiety": os.path.join(MODELS_DIR, "anxiety"),
    "bipolar": os.path.join(MODELS_DIR, "bipolar"),
    "depression": os.path.join(MODELS_DIR, "depression"),
    "ocd": os.path.join(MODELS_DIR, "ocd"),
    "schizophrenia": os.path.join(MODELS_DIR, "schizophrenia"),
}

DEVICE = torch.device("cpu")

NUM_LM = 2  # number of base copies

# How adapters are placed on each LM instance
# Keys: lm_index (0..NUM_LM-1); values: list of adapter names
# LM_ADAPTER_PLACEMENT = {
#     0: ["anxiety", "bipolar", "depression", "ocd", "schizophrenia"],   # all experts on single LM
# }
LM_ADAPTER_PLACEMENT = {
    0: ["anxiety", "bipolar", "depression"],   # "mood" models
    1: ["ocd", "schizophrenia", "depression"]  # "thought" models + a shared one
}

ENABLE_BATCHING = True
MAX_BATCH_SIZE = 4
MAX_BATCH_WAIT_S = 0.1

LOG_FILE_SERVER = "./logs/server_dual_lm__testing_logs.csv"
HF_TOKEN = "" # your HF token here with access to LLama-3.2-1B-Instruct

# =============================
# Logging infra
# =============================

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
        "chosen_lm_idx",
        "batch_size",
        "orchestrator_time_ms",
        "preprocess_time_ms",
        "queue_time_ms",
        "inference_time_ms",
        "total_time_ms",
    ],
)
server_logger.start()


def log_server(record: Dict):
    server_log_queue.put(record)


# =============================
# Orchestrator
# =============================

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


# =============================
# LM instances
# =============================

class LMInstance:
    def __init__(self, idx: int, base_path: str, tokenizer: AutoTokenizer, adapter_names: List[str]):
        self.idx = idx
        self.tokenizer = tokenizer
        self.active_count = 0
        self.active_lock = threading.Lock()
        self.semaphore = asyncio.Semaphore(1)  # to serialize set_adapter + generate

        print(f"[LM{idx}] Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_path,
            torch_dtype=torch.float32,
            device_map={"": DEVICE},
            token=HF_TOKEN,
        )

        if not adapter_names:
            raise ValueError(f"LM{idx} must have at least one adapter.")
        first_name = adapter_names[0]
        first_path = EXPERT_ADAPTERS[first_name]

        print(f"[LM{idx}] Loading first adapter '{first_name}' from {first_path}")
        self.model = PeftModel.from_pretrained(
            base_model,
            first_path,
            adapter_name=first_name,
        ).to(DEVICE)

        for name in adapter_names[1:]:
            path = EXPERT_ADAPTERS[name]
            print(f"[LM{idx}] Loading additional adapter '{name}' from {path}")
            self.model.load_adapter(path, adapter_name=name)

        self.current_adapter = first_name
        self.adapters = set(adapter_names)

        self.request_queue: Optional[asyncio.Queue] = None
        self.worker_task: Optional[asyncio.Task] = None

    # ---------- load accounting ----------
    def inc_active(self):
        with self.active_lock:
            self.active_count += 1

    def dec_active(self):
        with self.active_lock:
            self.active_count = max(0, self.active_count - 1)

    def get_load(self) -> int:
        with self.active_lock:
            return self.active_count

    # ---------- generate ----------
    async def generate(self, adapter_name: str, prompts: List[str], max_new_tokens: int = 128) -> List[str]:
        if adapter_name not in self.adapters:
            raise ValueError(f"LM{self.idx} does not have adapter '{adapter_name}'")

        async with self.semaphore:
            if adapter_name != self.current_adapter:
                self.model.set_adapter(adapter_name)
                self.current_adapter = adapter_name

            conversations = [[{"role": "user", "content": p}] for p in prompts]
            input_ids = self.tokenizer.apply_chat_template(
                conversations,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_generation_prompt=True,
            ).to(DEVICE)

            attn_mask = (input_ids != self.tokenizer.pad_token_id).long().to(DEVICE)
            input_len = input_ids.shape[1]

            def _do_generate():
                with torch.no_grad():
                    out_ids = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.9,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                gen_only = out_ids[:, input_len:]
                texts = self.tokenizer.batch_decode(gen_only, skip_special_tokens=True)
                return [t.strip() for t in texts]

            return await asyncio.to_thread(_do_generate)


def choose_lm_for_adapter(adapter_name: str, lm_pool: List[LMInstance]) -> int:
    candidates = [lm for lm in lm_pool if adapter_name in lm.adapters]
    if not candidates:
        raise RuntimeError(f"No LM has adapter '{adapter_name}'")
    best = min(candidates, key=lambda lm: lm.get_load())
    return best.idx


# =============================
# Batching layer
# =============================

@dataclass
class BatchJob:
    request_id: str
    func_id: int
    prompt: str
    enqueue_time: float
    adapter_name: str
    future: asyncio.Future


def elapsed_ms(a: float, b: float) -> float:
    return (b - a) * 1000.0


async def lm_batch_worker(lm: LMInstance):
    """Per-LM batching worker."""
    assert lm.request_queue is not None
    q = lm.request_queue

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

        jobs_by_adapter: Dict[str, List[BatchJob]] = {}
        for j in batch:
            jobs_by_adapter.setdefault(j.adapter_name, []).append(j)

        for adapter_name, jobs in jobs_by_adapter.items():
            batch_size = len(jobs)
            batch_start = time.perf_counter()

            lm.inc_active()
            try:
                inf_start = time.perf_counter()
                prompts = [j.prompt for j in jobs]
                outputs = await lm.generate(adapter_name, prompts)
                inf_end = time.perf_counter()
            finally:
                lm.dec_active()

            for j, out in zip(jobs, outputs):
                queue_time_ms = elapsed_ms(j.enqueue_time, batch_start)
                inf_time_ms = elapsed_ms(inf_start, inf_end)
                if not j.future.done():
                    j.future.set_result((out, queue_time_ms, inf_time_ms, batch_size))

        q.task_done()


# =============================
# FastAPI schemas
# =============================

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
    chosen_lm_idx: int
    timings_ms: Dict[str, float]
    batch_size: int


# =============================
# App setup
# =============================

app = FastAPI()

orchestrator: OrchestratorRouter
lm_pool: List[LMInstance] = []


@app.on_event("startup")
async def startup_event():
    global orchestrator, lm_pool

    orchestrator = OrchestratorRouter(ORCHESTRATOR_MODEL_PATH)

    tokenizer = AutoTokenizer.from_pretrained(BASE_LM_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # build LM pool
    for i in range(NUM_LM):
        adapter_names = LM_ADAPTER_PLACEMENT.get(i, [])
        if not adapter_names:
            raise ValueError(f"LM index {i} has no adapters configured.")
        lm = LMInstance(i, BASE_MODEL_PATH, tokenizer, adapter_names)
        lm_pool.append(lm)

    if ENABLE_BATCHING:
        loop = asyncio.get_event_loop()
        for lm in lm_pool:
            lm.request_queue = asyncio.Queue()
            lm.worker_task = loop.create_task(lm_batch_worker(lm))


# =============================
# Main endpoint
# =============================

@app.post("/infer", response_model=InferenceResponse)
async def infer(req: InferenceRequest):
    t0 = time.perf_counter()
    preprocess_start = t0

    # orchestrator
    orch_start = time.perf_counter()
    label = orchestrator.route(req.prompt)  # adapter_name
    orch_end = time.perf_counter()
    preprocess_end = orch_end

    lm_idx = choose_lm_for_adapter(label, lm_pool)
    lm = lm_pool[lm_idx]

    if ENABLE_BATCHING:
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        enqueue_time = time.perf_counter()
        job = BatchJob(
            request_id=req.request_id,
            func_id=req.func_id,
            prompt=req.prompt,
            enqueue_time=enqueue_time,
            adapter_name=label,
            future=fut,
        )
        await lm.request_queue.put(job)

        output, queue_time_ms, inf_time_ms, batch_size = await fut
    else:
        queue_time_ms = 0.0
        batch_size = 1

        lm.inc_active()
        try:
            inf_start = time.perf_counter()
            outputs = await lm.generate(label, [req.prompt])
            inf_end = time.perf_counter()
        finally:
            lm.dec_active()

        output = outputs[0]
        inf_time_ms = elapsed_ms(inf_start, inf_end)

    t_end = time.perf_counter()

    timings = {
        "orchestrator_time_ms": elapsed_ms(orch_start, orch_end),
        "preprocess_time_ms": elapsed_ms(preprocess_start, preprocess_end),
        "queue_time_ms": queue_time_ms,
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
            "chosen_lm_idx": lm_idx,
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
        chosen_lm_idx=lm_idx,
        timings_ms=timings,
        batch_size=batch_size,
    )

# run with:
#   python -m uvicorn server_dual_lm:app --host 0.0.0.0 --port 8000
