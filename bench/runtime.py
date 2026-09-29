"""One model and one execution stream/context per process."""
import time
import json

import numpy as np
import torch

from .model import ARTIFACTS, load_model


class TorchRuntime:
    def __init__(self, precision="fp16"):
        self.model = load_model("cuda")
        self.precision = precision
        self.stream = torch.cuda.Stream()

    @torch.inference_mode()
    def infer(self, signal, lengths):
        with torch.cuda.stream(self.stream):
            x = torch.from_numpy(signal).to("cuda")
            n = torch.from_numpy(lengths).to("cuda")
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.precision == "fp16"):
                age, gender = self.model(x, n)
            outputs = age.float().cpu().numpy(), gender.float().cpu().numpy()
        self.stream.synchronize()
        return outputs


class TrtRuntime:
    def __init__(self, engine):
        import tensorrt as trt
        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Could not load {engine}")
        self.context = self.engine.create_execution_context(self.trt.ExecutionContextAllocationStrategy.USER_MANAGED)
        self.workspace = None
        self.stream = torch.cuda.Stream()
        self.profile_index = -1
        self.profiles = [self.engine.get_tensor_profile_shape("signal", i) for i in range(self.engine.num_optimization_profiles)]

    def infer(self, signal, lengths):
        with torch.cuda.stream(self.stream):
            shape = signal.shape
            candidates = [i for i, (low, _, high) in enumerate(self.profiles)
                          if all(a <= b <= c for a, b, c in zip(low, shape, high))]
            if not candidates:
                raise RuntimeError(f"Shape {shape} exceeds engine profiles {self.profiles}")
            profile = candidates[0]
            if profile != self.profile_index:
                if not self.context.set_optimization_profile_async(profile, self.stream.cuda_stream):
                    raise RuntimeError(f"Cannot select TensorRT profile {profile}")
                self.profile_index = profile
            tensors = {"signal": torch.from_numpy(signal).to("cuda"),
                       "lengths": torch.from_numpy(lengths).to("cuda")}
            for name, value in tensors.items():
                if not self.context.set_input_shape(name, tuple(value.shape)):
                    raise ValueError(f"Unsupported shape {name}: {value.shape}")
            required = self.context.update_device_memory_size_for_shapes()
            if self.workspace is None or self.workspace.numel() < required:
                self.stream.synchronize()
                self.workspace = None
                self.workspace = torch.empty(required, dtype=torch.uint8, device="cuda")
            if required:
                self.context.device_memory = self.workspace.data_ptr()
            for name in ("age", "gender"):
                shape = tuple(self.context.get_tensor_shape(name))
                dtype = torch.from_numpy(np.empty((), dtype=self.trt.nptype(self.engine.get_tensor_dtype(name)))).dtype
                tensors[name] = torch.empty(shape, dtype=dtype, device="cuda")
            for name, tensor in tensors.items():
                self.context.set_tensor_address(name, tensor.data_ptr())
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError("TensorRT execution failed")
            outputs = tuple(tensors[name].float().cpu().numpy() for name in ("age", "gender"))
        self.stream.synchronize()
        return outputs


def export():
    model = load_model(attention="eager")
    x = torch.randn(2, 16000)
    lengths = torch.tensor([16000, 12000], dtype=torch.int64)
    path = ARTIFACTS / "batched.onnx"
    with torch.inference_mode():
        torch.onnx.export(model, (x, lengths), str(path), dynamo=False,
                          input_names=["signal", "lengths"], output_names=["age", "gender"],
                          dynamic_axes={"signal": {0: "batch", 1: "samples"},
                                        "lengths": {0: "batch"}, "age": {0: "batch"},
                                        "gender": {0: "batch"}}, opset_version=17)
    print(f"Exported {path}")


def build(seconds=30.0, max_batch=48, workspace_gib=2, audio_budget=240):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(ARTIFACTS / "batched.onnx")):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gib << 30)
    config.set_flag(trt.BuilderFlag.FP16)
    # FP32 reductions avoid the overflow encountered in FP16 layer normalization.
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if any(s in layer.name for s in ("layer_norm", "layernorm")):
            if layer.type not in (trt.LayerType.SHAPE, trt.LayerType.CONSTANT):
                layer.precision = trt.float32
    profiles = []
    for duration in (1, 3, 5, 10, 20, 30):
        if duration > seconds:
            continue
        cap = min(max_batch, audio_budget // duration) if audio_budget else max_batch
        if cap < 1:
            raise ValueError("Audio budget too small for a single recording")
        samples = duration * 16000
        profile = builder.create_optimization_profile()
        profile.set_shape("signal", (1, samples), (min(8, cap), samples), (cap, samples))
        profile.set_shape("lengths", (1,), (min(8, cap),), (cap,))
        config.add_optimization_profile(profile)
        profiles.append({"seconds": duration, "max_batch": cap})
    cache_path = ARTIFACTS / "timing.cache"
    cache = config.create_timing_cache(cache_path.read_bytes() if cache_path.exists() else b"")
    config.set_timing_cache(cache, ignore_mismatch=False)
    start = time.perf_counter()
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT engine build failed; inspect logged parser/precision errors")
    path = ARTIFACTS / f"fp16-{seconds:g}s-b{max_batch}.engine"
    path.write_bytes(bytes(plan))
    path.with_suffix(".json").write_text(json.dumps({"profiles": profiles, "audio_budget": audio_budget,
        "workspace_gib": workspace_gib, "build_seconds": time.perf_counter() - start,
        "tensorrt_version": trt.__version__}, indent=2) + "\n")
    cache_path.write_bytes(bytes(config.get_timing_cache().serialize()))
    print(f"Built {path} in {time.perf_counter() - start:.1f}s")
