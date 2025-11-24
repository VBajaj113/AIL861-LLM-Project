import asyncio
import csv
import httpx
import json
import os
import queue
import random
import threading
import time
from dataclasses import dataclass
from typing import Dict, List

TRACE_FILE = "./client_traces/trace1.csv"
PROMPTS_FILE = "./client_traces/prompts.json"
CLIENT_LOG_FILE = "./logs/client_logs_azure.csv"
SERVER_URL = "http://localhost:8000/infer"


# ============================================================
# Logging
# ============================================================

class ClientCsvLogger(threading.Thread):
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


client_log_queue = queue.Queue()
client_logger = ClientCsvLogger(
    client_log_queue,
    CLIENT_LOG_FILE,
    fieldnames=[
        "client_send_ts",
        "request_id",
        "func_id",
        "trace_start_time",
        "prompt",
        "response",
        "response_label",
        "response_expert",
        "e2e_time_ms",
        "server_total_time_ms",
        "batch_size",
    ],
)
client_logger.start()


def log_client(record):
    client_log_queue.put(record)


# ============================================================
# Trace + prompts
# ============================================================

@dataclass
class TraceEntry:
    request_id: str
    func_id: str
    start_time: float
    model_id: int


def load_trace(path: str) -> List[TraceEntry]:
    entries = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            entries.append(
                TraceEntry(
                    request_id=str(i),
                    func_id=row["func"],
                    start_time=float(row["startTime"]),
                    model_id=0
                )
            )
    entries.sort(key=lambda x: x.start_time)
    unique_funcs = set(e.func_id for e in entries)
    func_id_map = {func_id: i % 5 for i, func_id in enumerate(sorted(unique_funcs))}

    for entry in entries:
        entry.model_id = func_id_map[entry.func_id]
    return entries


def load_prompts(path: str) -> Dict[int, List[str]]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


# ============================================================
# Worker per trace entry
# ============================================================

async def send_request(
    client: httpx.AsyncClient,
    entry: TraceEntry,
    t0: float,
    first_start: float,
    prompts_by_func: Dict[int, List[str]],
):
    # respect relative timing from trace
    relative_start = entry.start_time - first_start
    now = time.perf_counter()
    delay = relative_start - (now - t0)
    if delay > 0:
        await asyncio.sleep(delay)

    prompts = prompts_by_func.get(entry.model_id)
    if not prompts:
        raise ValueError(f"Forgot to configure prompts for {entry.model_id} :p")
    prompt = random.choice(prompts)

    send_ts = time.time()
    t_start = time.perf_counter()

    payload = {
        "request_id": entry.request_id,
        "func_id": entry.model_id,
        "start_time": entry.start_time,
        "prompt": prompt,
    }

    resp = await client.post(SERVER_URL, json=payload, timeout=None)
    resp.raise_for_status()
    data = resp.json()

    t_end = time.perf_counter()
    e2e_ms = (t_end - t_start) * 1000.0

    timings = data.get("timings_ms", {})
    server_total = timings.get("total_time_ms", None)
    batch_size = data.get("batch_size", 1)

    log_client(
        {
            "client_send_ts": send_ts,
            "request_id": entry.request_id,
            "func_id": entry.model_id,
            "trace_start_time": entry.start_time,
            "prompt": prompt,
            "response_label": data.get("chosen_label"),
            "response_expert": data.get("chosen_expert"),
            "e2e_time_ms": e2e_ms,
            "server_total_time_ms": server_total,
            "batch_size": batch_size,
            "response": data.get("output"),
        }
    )

    print(
        f"[{entry.request_id}] func={entry.model_id} "
        f"label={data.get('chosen_label')} expert={data.get('chosen_expert')} "
        f"batch={batch_size} e2e={e2e_ms:.1f}ms server={server_total:.1f}ms"
    )


# ============================================================
# Main
# ============================================================

async def main():
    trace_entries = load_trace(TRACE_FILE)
    if not trace_entries:
        print("Trace is empty.")
        return

    prompts_by_func = load_prompts(PROMPTS_FILE)

    first_start = trace_entries[0].start_time
    t0 = time.perf_counter()

    async with httpx.AsyncClient() as client:
        tasks = [
            asyncio.create_task(
                send_request(client, entry, t0, first_start, prompts_by_func)
            )
            for entry in trace_entries
        ]
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
