"""Closed-loop clients; latency includes queueing, packing and GPU transfers."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from datetime import datetime, timezone
import json
import itertools
import importlib.metadata
import platform
import random
import subprocess
import threading
import time

import numpy as np
import torch

from .model import ROOT
from .prepare import audio_corpus, bucket_samples, pack, reference_workload, sha256
from .runtime import TorchRuntime, TrtRuntime


class GpuMonitor:
    """NVML samples include native TensorRT allocations, unlike torch counters."""
    def __init__(self):
        import pynvml
        self.nvml = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.collect, daemon=True)

    def collect(self):
        while not self.stop_event.is_set():
            self.samples.append({"used_mib": self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used / 2**20,
                                 "gpu_util_percent": self.nvml.nvmlDeviceGetUtilizationRates(self.handle).gpu,
                                 "temperature_c": self.nvml.nvmlDeviceGetTemperature(self.handle, 0)})
            self.stop_event.wait(0.05)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join()
        self.nvml.nvmlShutdown()

    def summary(self):
        return {"gpu_peak_used_mib": max(s["used_mib"] for s in self.samples),
                "gpu_mean_util_percent": float(np.mean([s["gpu_util_percent"] for s in self.samples])),
                "gpu_max_temperature_c": max(s["temperature_c"] for s in self.samples)}


def percentiles(values):
    return {f"p{p}": float(np.percentile(values, p)) for p in (50, 95, 99)}


async def trial(runtime, clips, batch_size, concurrency, duration, delay_ms, warmup, audio_budget=240):
    # Normalize/pack inside measured worker, unlike disk decoding (excluded).
    def execute(items):
        start = time.perf_counter()
        signal, lengths = pack(items, bucket=True)
        age, gender = runtime.infer(signal, lengths)
        if not np.isfinite(age).all() or not np.isfinite(gender).all():
            raise RuntimeError("Non-finite model output")
        return time.perf_counter() - start

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="single-model") as executor:
        loop = asyncio.get_running_loop()
        buckets = sorted(set(bucket_samples(len(clip)) for clip in clips))
        def limit(bucket):
            return min(batch_size, int(audio_budget * 16000 // bucket)) if audio_budget else batch_size
        for _ in range(warmup):
            for bucket in buckets:
                clip = next(c for c in clips if bucket_samples(len(c)) == bucket)
                warm_batch = min(limit(bucket), max(1, concurrency // len(buckets)))
                await loop.run_in_executor(executor, execute, [clip] * warm_batch)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        queue = asyncio.Queue(maxsize=concurrency)
        latencies, waits, batches, service = [], [], [], []
        raw_requests = []
        order = list(clips)
        random.Random(42).shuffle(order)
        source = itertools.cycle(order)
        start = time.perf_counter()
        deadline = start + duration

        async def worker():
            pending = deque()
            while True:
                first = pending.popleft() if pending else await queue.get()
                items = [first]
                bucket = bucket_samples(len(first[0]))
                batch_limit = limit(bucket)
                # Oldest pending request selects the bucket; defer incompatible
                # lengths without starving them or padding every request to 30s.
                for _ in range(len(pending)):
                    item = pending.popleft()
                    if bucket_samples(len(item[0])) == bucket and len(items) < batch_limit:
                        items.append(item)
                    else:
                        pending.append(item)
                batch_deadline = time.perf_counter() + delay_ms / 1000
                while len(items) < batch_limit:
                    try:
                        item = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        remaining = batch_deadline - time.perf_counter()
                        if remaining <= 0:
                            break
                        try:
                            item = await asyncio.wait_for(queue.get(), remaining)
                        except asyncio.TimeoutError:
                            break
                    if bucket_samples(len(item[0])) == bucket:
                        items.append(item)
                    else:
                        pending.append(item)
                dispatch = time.perf_counter()
                try:
                    elapsed = await loop.run_in_executor(executor, execute, [item[0] for item in items])
                    end = time.perf_counter()
                    batches.append(len(items))
                    service.append(elapsed * 1000)
                    for clip, submitted, future in items:
                        latency = (end - submitted) * 1000
                        wait = (dispatch - submitted) * 1000
                        latencies.append(latency)
                        waits.append(wait)
                        raw_requests.append([submitted - start, end - start, latency, wait, len(clip)])
                        future.set_result(None)
                except Exception as exc:
                    for _, _, future in items:
                        future.set_exception(exc)
                    # Propagate queued requests too so client tasks cannot hang.
                    while not queue.empty():
                        queue.get_nowait()[2].set_exception(exc)
                    while pending:
                        pending.popleft()[2].set_exception(exc)
                    raise

        async def client(index):
            while time.perf_counter() < deadline:
                future = loop.create_future()
                clip = next(source)
                await queue.put((clip, time.perf_counter(), future))
                await future

        task = asyncio.create_task(worker())
        clients = [asyncio.create_task(client(i)) for i in range(concurrency)]
        try:
            await asyncio.gather(*clients)
        finally:
            for client_task in clients:
                client_task.cancel()
            task.cancel()
            await asyncio.gather(task, *clients, return_exceptions=True)
        elapsed = time.perf_counter() - start
    if not latencies:
        raise RuntimeError("No completed requests")
    return {"status": "ok", "requests": len(latencies), "elapsed_s": elapsed,
            "rps": len(latencies) / elapsed,
            "audio_seconds_per_second": sum(x[4] / 16000 for x in raw_requests) / elapsed,
            "latency_ms": percentiles(latencies), "queue_ms": percentiles(waits),
            "batch_service_ms": percentiles(service), "actual_batch_mean": float(np.mean(batches)),
            "actual_batch_max": max(batches),
            "bucket_counts": {str(n / 16000): sum(bucket_samples(r[4]) == n for r in raw_requests) for n in buckets},
            "latency_by_bucket_ms": {str(n / 16000): percentiles([r[2] for r in raw_requests if bucket_samples(r[4]) == n])
                                     for n in buckets if any(bucket_samples(r[4]) == n for r in raw_requests)},
            "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "raw_requests": raw_requests}


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run on the GPU host, outside the restricted sandbox")
    torch.set_num_threads(args.cpu_threads)
    validation_path = ROOT / "artifacts" / ("validation.json" if args.backend == "torch" else "validation-trt.json")
    if not validation_path.exists() or not json.loads(validation_path.read_text())["passed"]:
        raise RuntimeError("Run numerical validation successfully before benchmarking this backend")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    clips, audio_manifest = reference_workload(args.reference_audio) if args.reference_audio else audio_corpus(args.audio_dir)
    runtime = TorchRuntime() if args.backend == "torch" else TrtRuntime(args.engine)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / "results" / f"{stamp}-{args.backend}"
    output.mkdir(parents=True)
    metadata = {"started_utc": stamp, "python": platform.python_version(),
                "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
                "gpu_capability": torch.cuda.get_device_capability(),
                "device_total_mib": torch.cuda.get_device_properties(0).total_memory / 2**20,
                "nvidia_smi": subprocess.check_output(["nvidia-smi"], text=True),
                "config": {key: str(value) if hasattr(value, "__fspath__") else value for key, value in vars(args).items()},
                "single_model_instances": 1, "single_execution_contexts": 1,
                "input_samples": sorted(set(map(len, clips))), "recordings": len(clips),
                "audio_manifest": audio_manifest,
                "measurement": "closed-loop, queue+normalization+padding+H2D+inference+D2H; excludes disk/audio decoding and HTTP",
                "versions": {name: importlib.metadata.version(name) for name in
                             ("transformers", "onnx", "onnxruntime", "tensorrt-cu13", "numpy", "nvidia-ml-py")},
                "code_sha256": {p.name: sha256(p) for p in (ROOT / "bench").glob("*.py")},
                "validation": json.loads(validation_path.read_text()),
                "manifest": json.loads((ROOT / "artifacts/manifest.json").read_text())}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    combinations = [(b, c, r) for r in range(args.repeats) for b in args.batches for c in args.concurrency]
    random.Random(args.seed).shuffle(combinations)
    for b, c, repeat in combinations:
        row = {"backend": args.backend, "batch_size": b, "concurrency": c, "repeat": repeat}
        try:
            torch.cuda.empty_cache()
            with GpuMonitor() as monitor:
                row.update(asyncio.run(trial(runtime, clips, b, c, args.duration, args.delay_ms, args.warmup, args.audio_budget)))
            row.update(monitor.summary())
        except (RuntimeError, MemoryError) as exc:
            row.update(status="failed", error=str(exc))
            torch.cuda.empty_cache()
        raw = row.pop("raw_requests", [])
        (output / f"b{b}-c{c}-r{repeat}.json").write_text(json.dumps({**row, "raw_requests": raw}) + "\n")
        with (output / "trials.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    report(output)
    print(f"Results: {output}", flush=True)


def report(output):
    rows = [json.loads(line) for line in (output / "trials.jsonl").read_text().splitlines()]
    groups = {}
    for row in rows:
        groups.setdefault((row["batch_size"], row["concurrency"]), []).append(row)
    summary = []
    for (b, c), trials in sorted(groups.items()):
        good = [r for r in trials if r["status"] == "ok"]
        record = {"batch_size": b, "concurrency": c, "successful_repeats": len(good), "failed_repeats": len(trials) - len(good)}
        if good:
            record.update(rps=float(np.median([r["rps"] for r in good])),
                          p50_ms=float(np.median([r["latency_ms"]["p50"] for r in good])),
                          p95_ms=float(np.median([r["latency_ms"]["p95"] for r in good])),
                          p99_ms=float(np.median([r["latency_ms"]["p99"] for r in good])),
                          actual_batch_mean=float(np.mean([r["actual_batch_mean"] for r in good])),
                          gpu_peak_mib=max(r["gpu_peak_used_mib"] for r in good))
        summary.append(record)
    import csv
    fields = ["batch_size", "concurrency", "successful_repeats", "failed_repeats", "rps", "p50_ms", "p95_ms", "p99_ms", "actual_batch_mean", "gpu_peak_mib"]
    with (output / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    good = [r for r in summary if "rps" in r and not r["failed_repeats"]]
    lines = ["# Single-instance benchmark", "", "Closed-loop local scheduler benchmark. Latency includes queueing, normalization, padding and GPU transfers; excludes HTTP and audio decoding. RPS is completed audio requests per second. Repeats are summarized by medians; raw request latencies are retained.", ""]
    if good:
        throughput = max(good, key=lambda r: r["rps"])
        latency = min(good, key=lambda r: r["p95_ms"])
        lines += [f"Highest RPS: batch cap {throughput['batch_size']}, concurrency {throughput['concurrency']}: {throughput['rps']:.2f} RPS, p95 {throughput['p95_ms']:.2f} ms.", "",
                  f"Lowest p95: batch cap {latency['batch_size']}, concurrency {latency['concurrency']}: {latency['rps']:.2f} RPS, p95 {latency['p95_ms']:.2f} ms.", ""]
    lines += ["| Batch cap | Concurrency | RPS | p50 ms | p95 ms | p99 ms | Actual mean batch | Peak GPU MiB | Failed repeats |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        values = [str(r["batch_size"]), str(r["concurrency"])] + [f"{r[k]:.2f}" if k in r else "—" for k in ("rps", "p50_ms", "p95_ms", "p99_ms", "actual_batch_mean", "gpu_peak_mib")] + [str(r["failed_repeats"])]
        lines.append("| " + " | ".join(values) + " |")
    lines += ["", "A batch cap is a maximum, not a guaranteed realized batch. Concurrency below that cap cannot fill it. Peak GPU memory is whole-device NVML sampling every 50 ms, including warm-up; brief peaks can be missed. Queue delay is only the batch collection window; total waiting can exceed it under load.", ""]
    (output / "REPORT.md").write_text("\n".join(lines))
