import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from huggingface_hub import hf_hub_download

from .model import ARTIFACTS, CHECKPOINT, ROOT, SAMPLE_RATE

REPO = "audeering/wav2vec2-large-robust-6-ft-age-gender"
REVISION = "a681b720dafd12b9dd7b6d13fb437c7b6b197fd3"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare():
    CHECKPOINT.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "preprocessor_config.json", "model.safetensors"):
        hf_hub_download(REPO, name, revision=REVISION, local_dir=CHECKPOINT)
    manifest = {"model": REPO, "revision": REVISION,
                "weights_sha256": sha256(CHECKPOINT / "model.safetensors"),
                "original_onnx_sha256": sha256(ROOT / "model/model.onnx")}
    (ARTIFACTS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def audio_corpus(directory, max_seconds=30, validation_windows=False):
    """Read user recordings as-is; never synthesize, repeat, trim or save audio."""
    clips, manifest = [], []
    paths = sorted(p for p in directory.rglob("*") if p.suffix.lower() in {".wav", ".flac", ".ogg", ".mp3"})
    if not paths:
        raise RuntimeError(f"No recordings in {directory}. Add WAV/FLAC files, preferably varied durations up to {max_seconds}s.")
    for path in paths:
        info = sf.info(path)
        if info.duration > max_seconds + 1 / info.samplerate and not validation_windows:
            raise ValueError(f"{path.name}: {info.duration:.2f}s exceeds {max_seconds}s. Supply a shorter recording.")
        audio, sr = sf.read(path, dtype="float32", frames=int(max_seconds * info.samplerate) if validation_windows else -1)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            from math import gcd
            divisor = gcd(sr, SAMPLE_RATE)
            audio = resample_poly(audio, SAMPLE_RATE // divisor, sr // divisor).astype(np.float32)
        if len(audio) < 400 or not np.isfinite(audio).all():
            raise ValueError(f"{path.name}: too short or contains non-finite audio samples")
        clips.append(audio)
        manifest.append({"file": str(path.relative_to(directory)), "sha256": sha256(path),
                         "seconds": len(audio) / SAMPLE_RATE, "original_sample_rate": sr})
    return clips, manifest


def bucket_samples(samples):
    for seconds in (1, 3, 5, 10, 20, 30):
        if samples <= seconds * SAMPLE_RATE:
            return seconds * SAMPLE_RATE
    raise ValueError("Audio exceeds 30 seconds")


def reference_workload(path):
    """Deterministic windows of existing speech, held only in memory."""
    info = sf.info(path)
    audio, sr = sf.read(path, frames=info.samplerate * 180, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        from math import gcd
        divisor = gcd(sr, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // divisor, sr // divisor).astype(np.float32)
    clips, manifest = [], []
    digest = sha256(path)
    for seconds in (1, 3, 5, 10, 20, 30):
        samples = seconds * SAMPLE_RATE
        if len(audio) < samples:
            raise ValueError(f"Reference needs at least {seconds} seconds of speech")
        for offset in np.linspace(0, len(audio) - samples, 8, dtype=int):
            clips.append(audio[offset:offset + samples].copy())
            manifest.append({"file": str(path), "sha256": digest,
                             "offset_seconds": int(offset) / SAMPLE_RATE, "seconds": seconds,
                             "original_sample_rate": sr})
    return clips, manifest


def pack(clips, normalize=True, bucket=False):
    lengths = np.array([len(x) for x in clips], dtype=np.int64)
    width = bucket_samples(int(lengths.max())) if bucket else int(lengths.max())
    signal = np.zeros((len(clips), width), dtype=np.float32)
    for i, clip in enumerate(clips):
        if normalize:
            clip = (clip - clip.mean()) / np.sqrt(clip.var() + 1e-7)
        signal[i, :len(clip)] = clip
    return signal, lengths
