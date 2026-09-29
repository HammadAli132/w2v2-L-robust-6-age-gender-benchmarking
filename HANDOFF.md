# R&D handoff — age/gender inference

Last updated: 2026-09-25, after Canterville authorization. **Benchmark matrix COMPLETE (both backends, 120 trials, 0 failures).** Remaining items are open discussion points with the user. Update this file after every substantial action and before stopping.

### Latest progress (read first)

- Final engine `artifacts/fp16-30s-b48.engine`: six duration profiles, caps 48/48/48/24/12/8 (240 audio-s budget), one context. `artifacts/validation-trt.json` passed=true (max 0.081 years / 0.0122 logits, incl. 30 s × 8 check).
- **Actual results** (Canterville reference workload, equal counts 1/3/5/10/20/30 s, 3 repeats × 15 s, medians):
  - TRT: `results/20260925T164035Z-trt/REPORT.md`. Throughput flat at ~200–215 RPS across all 20 configs (~2,450 audio-s/s). Highest RPS b16/c128 215.1 RPS (p95 828 ms); lowest p95 b32/c16 201.8 RPS, p95 150 ms. Peak NVML ~3.2 GiB (one outlier 7.27 GiB at b48/c64, unexplained, possibly transient/desktop).
  - Torch SDPA FP16: `results/20260925T165633Z-torch/REPORT.md`. ~75–78 RPS; lowest p95 b8/c16 402.7 ms; peak NVML ~6.9–7.9 GiB.
  - GPU is saturated already at c16: extra concurrency only adds queue latency. Realized batch means ≤16.9 because the six equal-frequency buckets split the queue; caps 32/48 are rarely filled.
- Smoke runs kept separately in `results/smoke/`. Full log `results/matrix.log`.
- Report written: `Age-Gender-Inference-Benchmark-Report.docx` (generator scripts were in the session scratchpad; tables are read from results/*.json|csv).
- TRT 7,273 MiB outlier explained: first trial after engine load; torch allocator reserved 6,404 MiB vs live 2,331 MiB (others reserved 2,338). The workspace grows stepwise in `TrtRuntime.infer`, and old buffers stay cached until `empty_cache()`. Proposed fix (NOT implemented, awaiting user): preallocate the max workspace over all profiles at startup.
- Open points raised with user (not implemented): scheduler bucket-selection policy (oldest vs largest group), workload realism (single narrator, uniform durations), whether to rerun with a production duration distribution.

## User objective and constraints

- Work in this same project directory; keep the environment organized and small.
- Use `uv add` for installing packages, never `pip` or `uv pip install`.
- One resident model / one execution context / one inference worker, no replicas.
- Benchmark maximum batch sizes **8, 16, 32, 48** against concurrent clients **16, 32, 48, 64, 128** (20 combinations).
- Report actual RPS and p50/p95/p99 latency, memory and failures. Do not invent results.
- Production recordings vary up to **30 seconds**. No p95 target was supplied.
- User initially offered to provide recordings, then explicitly authorized using **`data/reference/canterville.ogg`** to avoid delay. Use real excerpts in memory; do not synthesize or write new audio files.
- User requests this Markdown handoff so Claude or another agent can resume if limits are reached.

## Hardware and environment

- NVIDIA GeForce RTX 5060, 8151 MiB reported VRAM, compute capability 12.0 (`sm_120`).
- Driver 595.91.07; `nvidia-smi` reports CUDA compatibility 13.2.
- Installed PyTorch **2.15.0.dev20260925+cu130**, CUDA runtime 13.0; `sm_120` is in its compiled architectures. FP16 GPU matrix multiplication tested successfully.
- TensorRT **10.16.1.11** installed via `tensorrt-cu13`.
- Python 3.12.14, uv 0.12.10. Dependencies and exact resolution are in `pyproject.toml` / `uv.lock`.
- GPU devices are hidden in the Codex sandbox. GPU commands and uv cache/network access require `exec_command` with `sandbox_permissions=require_escalated`. This is a sandbox constraint; the actual driver works.
- Latest nightly resolution required both `torch` and its `triton` compiler dependency to use the official `https://download.pytorch.org/whl/nightly/cu130` index. `triton` here is the compiler package, **not NVIDIA Triton Inference Server**. No inference server/container is installed.
- `tool.uv.prerelease="allow"` and a Linux x86-64 required environment are persisted. A failed initial installation rolled back dependencies; a later successful `uv add` installed all required packages. `uv run --locked` currently works.

## Original project and model findings

- Original files: `notebookbef6cbf75b.ipynb`, `model/model.yaml`, `model/model.onnx` (347 MiB).
- `PROJECT_SUMMARY.md` was written earlier to explain the notebook and model.
- Model: audeering wav2vec2-large-robust, 7 convolution blocks, 6 transformer layers, hidden size 1024, 16 attention heads; temporal mean pooling; separate Linear→Tanh→Linear age (1 output) and gender (3 outputs) heads.
- Local ONNX accepts `[1, samples]` and contains waveform mean/variance normalization internally. Gender output is **raw logits**, ordered **female, male, child**. Age scalar ×100 gives years.
- Do not copy the upstream model-card wrapper unchanged: it adds softmax, unlike the local ONNX. Batch export also needs attention masking AND masked temporal pooling, plus normalization before padding.
- Pinned original PyTorch weights: `audeering/wav2vec2-large-robust-6-ft-age-gender`, revision `a681b720dafd12b9dd7b6d13fb437c7b6b197fd3`.
- Weights SHA256 `4a0f33264b690efcdee7d6cb803404486f9f93fd58355bbb0e6aa6f40db1e2e8`.
- Original ONNX SHA256 `75c5cc3debc2013215cee5f331a66b59bd7205da170fb61abab46fc4507df7be`.
- Original assets have not been modified.

## Implemented files

- `bench/model.py`: loads original safetensors strictly, remaps legacy weight_norm keys, supplies attention mask and masked temporal pooling, preserves raw heads.
- `bench/prepare.py`: downloads pinned model only; reads user audio and provenance; normalizes before padding; duration buckets 1/3/5/10/20/30 sec.
- `bench/runtime.py`: PyTorch SDPA+FP16 runtime; dynamic ONNX export; TensorRT builder and single-context runtime using user-managed activation memory.
- `bench/validate.py`: compares original ONNX against FP32/PyTorch, padded mixed batches, FP16, exported ONNX; optional TRT comparisons.
- `bench/benchmark.py`: asynchronous closed-loop clients, one inference thread, duration buckets, queueing, 1 ms batch collection window, randomized repeats, NVML memory monitoring, raw request times, JSONL/CSV/Markdown output.
- `bench/__main__.py`: CLI `prepare`, `export`, `build`, `validate`, `run`, `report`.
- `tests/test_benchmark.py`: 4 passing tests for normalization/padding, one-worker execution and accounting, duration grouping/concurrency cap, failure propagation without client hangs.
- `README.md`: usage and measurement definitions. It currently describes user-supplied data; update for the newly authorized Canterville workload.
- `.gitignore`: excludes virtualenv, cache, generated models, data and Python cache. Results should remain visible.

## Completed validation (actual measured numerical differences)

`artifacts/validation.json` is **passed=true**, checked with `--audio-dir data/reference`. These are numerical parity checks, not a performance benchmark or dataset-wide accuracy assessment.

| Comparison | Max age error (years) | Max gender logit error | Class agreement |
|---|---:|---:|---:|
| Original ONNX vs normalized FP32 PyTorch | 0.0003606 | 0.00003195 | 100% |
| Mixed padded vs individual FP32 | 0.00001192 | 0.000000954 | 100% |
| Long mixed vs individual FP32 (up to 30 s) | 0.00001192 | 0.000000954 | 100% |
| FP16 vs FP32, short | 0.027913 | 0.005042 | 100% |
| FP16 vs FP32, long | 0.005233 | 0.002971 | 100% |
| Exported ONNX vs FP32 | 0.0003427 | 0.00002909 | 100% |

Acceptance: <0.25 years, <0.03 raw gender logits, unchanged class on check samples. Raw unnormalized PyTorch differs by ~12 years, confirming normalization is essential.

## TensorRT progress / current issue

1. Export succeeded: `artifacts/batched.onnx`, opset 17, dynamic batch/sample dimensions and lengths input.
2. A broad profile max **48 × 30 s** failed building at the first convolution (no available tactic). Log: `artifacts/build.log`.
3. The smaller profile max **8 × 30 s** built successfully: `artifacts/fp16-30s-b8.engine` (~177 MiB), `artifacts/build-b8.log`, `artifacts/timing.cache`.
4. The 8-batch engine has **not yet been numerically validated**. The final matrix requires batch caps up to 48, so the engine setup still needs adjustment.

Likely broad-profile issue: first convolution at batch 48, 30 s produces ~2.36 billion elements, exceeding practical tensor/memory limits on this GPU. This is a hypothesis supported by the smaller successful build, not yet a complete engine diagnostic.

**Next implementation step:** use multiple duration-specific optimization profiles inside ONE TensorRT engine/context (not model replicas). Consider an explicit total padded-audio budget, e.g. 240 audio-seconds per batch: 1 s→48, 3 s→48, 5 s→48, 10 s→24, 20 s→12, 30 s→8. The scheduler should expose and record this policy rather than silently lower configured batch caps. Keep the user's 4 requested batch caps and report actual realized batch sizes. Validate and measure whether this budget fits; it is not yet an established optimum. Alternatively, keep strict caps and record unsupported/OOM configurations honestly.

## Audio currently on disk

- `data/reference/1.flac`: 10.435 s, 16 kHz.
- `data/reference/2.flac`: 3.275 s, 16 kHz.
- `data/reference/canterville.ogg`: 31 MB, ~4989 s, 44.1 kHz natural audiobook recording. User now explicitly authorizes this for performance tests.
- All were downloaded from `Narsil/asr_dummy` revision `8d141c84e3f84c54cd7bbaa851d24edd0f559734`. Canterville SHA256 `3a7c94d683543dd4fef0bebe12bcddbd302ffba5367a3280ecd602ffcf481e85`.
- No audio was synthesized or saved as generated excerpts. `data/audio/` is empty.
- **Current benchmark rejects Canterville because it rejects >30 s input files. Add an explicit reference-workload option that reads a finite source window and selects deterministic real 1/3/5/10/20/30 s excerpts IN MEMORY. Document the distribution and excerpt offsets in metadata.** Do not benchmark the full 83-minute file.

## Benchmark methodology to preserve

- Batch caps 8/16/32/48 × client concurrency 16/32/48/64/128.
- Defaults: 3 randomized repeats, 15 s submission window per trial, warmup per duration bucket, all submitted work drains before calculating RPS.
- Latency includes queue wait, normalization/padding, H2D, model execution, D2H and finite-output checks.
- Excludes disk/audio decoding, HTTP, model load and engine compile. Label as **local scheduler/inference benchmark**, not FastAPI/Triton end-to-end network results.
- Single model, single context/stream, single inference worker. Clients each have one outstanding request.
- Report p50/p95/p99, RPS, audio-seconds/sec, actual batches, latency by duration bucket, failures, whole-device NVML memory sampled every 50 ms. Save environment, validation, model/audio/code hashes, raw request times.
- Higher RPS and lower latency can select different configurations; don't invent one universal winner without an SLO.
- Run GPU comparisons sequentially (never compile and benchmark concurrently).

## Commands already successful

```bash
uv run --locked python -m unittest discover -s tests -v
uv run --locked python -m bench prepare
uv run --locked python -m bench export
uv run --locked python -m bench validate --audio-dir data/reference
uv run --locked python -m bench build --seconds 30 --max-batch 8
```

## To finish

1. Check/collect any previous tool process output; last successful build is the b8 engine above.
2. Add the authorized Canterville in-memory benchmark workload and update README.
3. Fix TensorRT profiles/runtime selection; declare resource budget clearly if used. Build a final engine supporting short batches up to 48 and recordings up to 30 s with a single context.
4. Validate the final engine on short batch sizes 1/8/16/32/48 and long mixed inputs, including profile switching. Fix numerical issues before benchmarking.
5. Run all 20 configurations with actual GPU measurements, repeat trials, save results under `results/`. PyTorch SDPA FP16 is a working baseline if TRT has problems.
6. Inspect results, summarize highest RPS and lowest latency with workload and limits, link reports. Do not call unexecuted code 'actual results'.
7. Update this handoff with commands, result paths, failures and final status after each milestone.
