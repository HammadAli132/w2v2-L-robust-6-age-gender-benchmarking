import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Single-instance age/gender inference R&D")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", help="Download pinned original model weights only")
    sub.add_parser("export", help="Export dynamic-batch, masked ONNX")
    build = sub.add_parser("build", help="Build one TensorRT FP16 engine")
    build.add_argument("--seconds", type=float, default=30)
    build.add_argument("--max-batch", type=int, default=48)
    build.add_argument("--workspace-gib", type=int, default=2)
    build.add_argument("--audio-budget", type=int, default=240)
    validate = sub.add_parser("validate", help="Compare original ONNX, padded batches and reduced precision")
    validate.add_argument("--engine", type=Path)
    validate.add_argument("--audio-dir", type=Path, default=Path("data/audio"))
    run = sub.add_parser("run", help="Run the batch/concurrency matrix")
    run.add_argument("--backend", choices=["torch", "trt"], default="torch")
    run.add_argument("--engine", type=Path, default=Path("artifacts/fp16-30s-b48.engine"))
    run.add_argument("--batches", nargs="+", type=int, default=[8, 16, 32, 48])
    run.add_argument("--concurrency", nargs="+", type=int, default=[16, 32, 48, 64, 128])
    run.add_argument("--audio-dir", type=Path, default=Path("data/audio"))
    run.add_argument("--reference-audio", type=Path, help="Use in-memory 1/3/5/10/20/30 s excerpts of this existing recording")
    run.add_argument("--audio-budget", type=int, default=240, help="Max total padded audio-seconds per batch; 0 disables (may OOM)")
    run.add_argument("--duration", type=float, default=15)
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--warmup", type=int, default=3)
    run.add_argument("--delay-ms", type=float, default=1)
    run.add_argument("--cpu-threads", type=int, default=4)
    run.add_argument("--seed", type=int, default=42)
    report = sub.add_parser("report", help="Regenerate report and CSV from saved trials")
    report.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        from .prepare import prepare
        prepare()
    elif args.command == "export":
        from .runtime import export
        export()
    elif args.command == "build":
        from .runtime import build
        build(args.seconds, args.max_batch, args.workspace_gib, args.audio_budget)
    elif args.command == "validate":
        from .validate import validate
        validate(args.engine, args.audio_dir)
    elif args.command == "run":
        from .benchmark import run
        if min(args.batches + args.concurrency) <= 0 or min(args.duration, args.repeats) <= 0:
            parser.error("Batch, concurrency, duration and repeats must be positive")
        if args.audio_budget != 0 and args.audio_budget < 30:
            parser.error("Audio budget must be 0 or at least 30 seconds")
        run(args)
    else:
        from .benchmark import report
        report(args.directory)


if __name__ == "__main__":
    main()
