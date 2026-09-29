# Age/gender inference R&D

One resident model, one GPU execution context, one inference worker. The original notebook and `model/` are retained as the reference. Dependencies are managed by `uv add` and pinned in `uv.lock`.

## Layout

- `bench/`: model wrapper, export, numerical validation and benchmark CLI.
- `model/`: original ONNX model and label configuration.
- `artifacts/`: downloaded checkpoint, dynamic ONNX, TensorRT engine, validation and provenance (generated).
- `data/audio/`: your benchmark recordings (provided manually).
- `data/reference/`: earlier downloaded public recordings, used only for conversion checks.
- `results/`: timestamped metadata, raw request timings, trial records, CSV and Markdown reports.

## Reproduce

Run from this directory on the GPU host. The NVIDIA device must be visible to the process.

```bash
uv sync --locked
uv run python -m bench prepare
uv run python -m bench export
uv run python -m bench validate --audio-dir data/audio
uv run python -m bench build --seconds 30 --max-batch 48
uv run python -m bench validate --engine artifacts/fp16-30s-b48.engine --audio-dir data/audio
uv run python -m bench run --backend torch
uv run python -m bench run --backend trt
```

The currently authorized reference run uses the existing Canterville recording:

```bash
uv run --locked python -m bench run --backend torch --reference-audio data/reference/canterville.ogg
uv run --locked python -m bench run --backend trt --reference-audio data/reference/canterville.ogg
```

This reads the first 180 seconds and selects eight deterministic excerpts at each of 1/3/5/10/20/30 seconds, in memory only. No audio files are generated. The input sequence has equal counts per duration; source hash and excerpt offsets are saved. This is one narrator, useful for performance testing but not accuracy evaluation.

All new packages must be installed with `uv add`. The PyTorch source is the CUDA 13.0 nightly index. The CUDA version printed by `nvidia-smi` is driver compatibility, not the version a Python wheel must exactly match.

## Experiment

The default matrix is maximum batch sizes **8, 16, 32, 48** × concurrent clients **16, 32, 48, 64, 128**. Each configuration has three randomized repeats, each submitting work for 15 seconds after three warm-up batches. All submitted requests drain before throughput is calculated. One client has at most one outstanding request; it immediately submits another upon completion until the trial deadline.

Place real recordings in `data/audio/` (or pass `--audio-dir`). WAV/FLAC is preferred; OGG/MP3 is also readable. Supply varied durations up to 30 seconds; a useful starting corpus is 20–50 recordings spanning the ranges below. More recordings improve representativeness; a smaller corpus is accepted. The loader never synthesizes, loops, concatenates, trims or saves benchmark audio. Recordings over 30 seconds are rejected. Decoding resamples to 16 kHz mono in memory.

Requests cycle through a deterministically shuffled list of your recordings, so each file has equal frequency in the input sequence. The measured duration distribution comes from your supplied files and is saved with their hashes. Length buckets have upper bounds of **1, 3, 5, 10, 20 and 30 seconds**. Each batch uses one bucket; the oldest waiting request selects it. Normalization excludes padding, and masks preserve valid sample lengths.

```bash
uv run python -m bench run --backend torch --duration 30 --repeats 3
uv run python -m bench run --backend trt --audio-dir data/audio
uv run python -m bench report results/RUN_DIRECTORY
```

The batch collection window defaults to 1 ms (`--delay-ms`). A configured batch size is a cap: concurrency 16 cannot populate a batch of 48. Reports include realized batch sizes. Higher concurrency can raise queue latency without increasing throughput.

An explicit **240 padded-audio-seconds per batch** resource budget (`--audio-budget 240`) applies identically to both backends. Effective batch cap is `min(configured cap, floor(240 / bucket_seconds))`: the maximum engine profile supports 48 at 1/3/5 s, 24 at 10 s, 12 at 20 s and 8 at 30 s. This keeps the large waveform feature maps bounded on 8 GB VRAM. Six duration profiles share one engine and execution context; they are not model replicas. Use `--audio-budget 0` for strict count-only experiments (which may exceed engine profiles or memory). The budget and actual batches are saved explicitly; these runs do not claim to execute 48 long recordings simultaneously.

Latency starts at request submission and ends when CPU outputs are available. It includes queueing, per-waveform normalization, padding, host/device transfers and inference. Audio decoding, file I/O, HTTP transport, model loading and engine compilation are excluded. This is a local scheduler benchmark, not a FastAPI/Triton network benchmark. It isolates the backend and scheduling policy without extra services.

RPS is successful audio requests per wall-clock second, including final drain time. Audio-seconds/second is also saved. Reports independently identify highest RPS and lowest p95 because there is no single optimum without a latency target. Raw latencies support further analysis. This closed-loop test is not an open-loop overload/capacity test.

## Numerical checks

The wrapper preserves the original heads and raw gender logits. It masks padding in transformer attention and temporal pooling. Validation reads short windows and up to 30 seconds from existing recordings in memory; it never writes audio files. It compares raw local ONNX inputs with normalized PyTorch inference, mixed padded batches with individual FP32 inference, exported ONNX, FP16 PyTorch and TensorRT at batch sizes 1/8/16/32/48 on short inputs, plus the longest available recording. Reduced precision must meet the recorded tolerances; this small smoke set does not establish dataset-wide accuracy. Before a full performance run, validate on your own corpus too.

Engines are local to the GPU/software configuration. No inference instances are replicated. Reports use whole-device NVML memory sampled every 50 ms, including warm-up; short peaks can be missed and desktop GPU use is included. PyTorch allocator counters are retained separately in raw trials and exclude TensorRT-owned allocations.
