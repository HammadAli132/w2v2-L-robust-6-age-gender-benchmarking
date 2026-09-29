# Single-instance benchmark

Closed-loop local scheduler benchmark. Latency includes queueing, normalization, padding and GPU transfers; excludes HTTP and audio decoding. RPS is completed audio requests per second. Repeats are summarized by medians; raw request latencies are retained.

Highest RPS: batch cap 48, concurrency 128: 79.47 RPS, p95 2411.47 ms.

Lowest p95: batch cap 48, concurrency 128: 79.47 RPS, p95 2411.47 ms.

| Batch cap | Concurrency | RPS | p50 ms | p95 ms | p99 ms | Actual mean batch | Peak GPU MiB | Failed repeats |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 48 | 128 | 79.47 | 1321.60 | 2411.47 | 2677.15 | 15.68 | 7765.44 | 0 |

A batch cap is a maximum, not a guaranteed realized batch. Concurrency below that cap cannot fill it. Peak GPU memory is whole-device NVML sampling every 50 ms, including warm-up; brief peaks can be missed. Queue delay is only the batch collection window; total waiting can exceed it under load.
