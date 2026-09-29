import json

import numpy as np
import onnxruntime as ort
import torch

from .model import ARTIFACTS, ROOT, load_model
from .prepare import audio_corpus, pack
from .runtime import TorchRuntime, TrtRuntime


def errors(actual, expected):
    return {"max_age_error_years": float(np.max(np.abs(actual[0] - expected[0])) * 100),
            "max_gender_logit_error": float(np.max(np.abs(actual[1] - expected[1]))),
            "gender_agreement": float(np.mean(actual[1].argmax(-1) == expected[1].argmax(-1)))}


def validate(engine=None, audio_dir=None):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    corpus, audio_manifest = audio_corpus(audio_dir or ROOT / "data/audio", validation_windows=True)
    clips = [clip[:min(len(clip), limit)] for clip in corpus[:2] for limit in (16000, 32000, 48000)]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(ROOT / "model/model.onnx"), options, providers=["CPUExecutionProvider"])
    model = load_model()
    checks = {}
    # Audit both raw and normalized inputs; local ONNX may normalize internally.
    with torch.inference_mode():
        raw_refs = [session.run(["logits_age", "logits_gender"], {"signal": x[None, :]}) for x in clips]
        expected = tuple(np.concatenate([y[i] for y in raw_refs]) for i in range(2))
        normalized = [model(*(torch.from_numpy(x) for x in pack([clip]))) for clip in clips]
        raw = [model(*(torch.from_numpy(x) for x in pack([clip], normalize=False))) for clip in clips]
        norm_values = tuple(np.concatenate([y[i].numpy() for y in normalized]) for i in range(2))
        raw_values = tuple(np.concatenate([y[i].numpy() for y in raw]) for i in range(2))
        checks["original_onnx_vs_normalized_torch"] = errors(norm_values, expected)
        checks["original_onnx_vs_raw_torch"] = errors(raw_values, expected)
        batched = model(*(torch.from_numpy(x) for x in pack(clips)))
        checks["mixed_padded_vs_individual_fp32"] = errors(tuple(x.numpy() for x in batched), norm_values)
    del model, session
    long_clips = [min(corpus, key=len), max(corpus, key=len)]
    fp32 = TorchRuntime(precision="fp32")
    individual_long = [fp32.infer(*pack([clip])) for clip in long_clips]
    long_reference = tuple(np.concatenate([y[i] for y in individual_long]) for i in range(2))
    checks["long_mixed_vs_individual_fp32"] = errors(fp32.infer(*pack(long_clips)), long_reference)
    del fp32
    torch.cuda.empty_cache()
    fp16 = TorchRuntime()
    checks["torch_fp16_vs_fp32"] = errors(fp16.infer(*pack(clips)), norm_values)
    checks["long_torch_fp16_vs_fp32"] = errors(fp16.infer(*pack(long_clips)), long_reference)
    del fp16
    torch.cuda.empty_cache()
    if (ARTIFACTS / "batched.onnx").exists():
        exported = ort.InferenceSession(str(ARTIFACTS / "batched.onnx"), options, providers=["CPUExecutionProvider"])
        signal, lengths = pack(clips)
        checks["exported_onnx_vs_fp32"] = errors(exported.run(None, {"signal": signal, "lengths": lengths}), norm_values)
        del exported
    if engine:
        runtime = TrtRuntime(engine)
        checks["tensorrt_fp16_vs_fp32"] = errors(runtime.infer(*pack(clips)), norm_values)
        checks["long_tensorrt_fp16_vs_fp32"] = errors(runtime.infer(*pack(long_clips)), long_reference)
        # Check extreme supported batches, not just batch 1.
        for batch in (1, 8, 16, 32, 48):
            sample = [clips[i % len(clips)] for i in range(batch)]
            reference = tuple(np.stack([norm_values[j][i % len(clips)] for i in range(batch)]) for j in range(2))
            checks[f"tensorrt_batch_{batch}"] = errors(runtime.infer(*pack(sample)), reference)
        # Longest recording at its budgeted profile cap; switches back from short profiles.
        cap = max(p[2][0] for p in runtime.profiles if p[2][1] >= len(long_clips[-1]))
        reference = tuple(np.repeat(long_reference[j][-1:], cap, axis=0) for j in range(2))
        checks[f"tensorrt_long_batch_{cap}"] = errors(runtime.infer(*pack([long_clips[-1]] * cap, bucket=True)), reference)
    # Practical numerical acceptance tolerances, not a dataset accuracy claim.
    passed = all(v["max_age_error_years"] < 0.25 and v["max_gender_logit_error"] < 0.03 and v["gender_agreement"] == 1
                 for key, v in checks.items() if key != "original_onnx_vs_raw_torch")
    result = {"passed": passed, "samples": len(clips), "checks": checks,
              "audio_manifest": audio_manifest,
              "longest_checked_seconds": max(map(len, long_clips)) / 16000,
              "tolerances": {"age_years": 0.25, "gender_logits": 0.03, "gender_agreement": 1.0}}
    path = ARTIFACTS / ("validation-trt.json" if engine else "validation.json")
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if not passed:
        raise RuntimeError("Numerical validation failed; do not trust benchmark results until resolved")
