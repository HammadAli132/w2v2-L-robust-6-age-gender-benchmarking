# Single-instance benchmark

Closed-loop local scheduler benchmark. Latency includes queueing, normalization, padding and GPU transfers; excludes HTTP and audio decoding. RPS is completed audio requests per second. Repeats are summarized by medians; raw request latencies are retained.

Highest RPS: batch cap 16, concurrency 64: 78.41 RPS, p95 1397.64 ms.

Lowest p95: batch cap 8, concurrency 16: 74.71 RPS, p95 402.65 ms.

| Batch cap | Concurrency | RPS | p50 ms | p95 ms | p99 ms | Actual mean batch | Peak GPU MiB | Failed repeats |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 16 | 74.71 | 213.30 | 402.65 | 414.46 | 3.98 | 6881.44 | 0 |
| 8 | 32 | 77.98 | 401.14 | 886.88 | 897.88 | 7.88 | 7655.44 | 0 |
| 8 | 48 | 77.91 | 614.76 | 1073.91 | 1100.65 | 7.85 | 7855.44 | 0 |
| 8 | 64 | 77.67 | 793.95 | 1145.90 | 1230.74 | 7.95 | 7855.44 | 0 |
| 8 | 128 | 78.00 | 1595.81 | 2117.27 | 2131.63 | 7.85 | 7855.44 | 0 |
| 16 | 16 | 74.68 | 212.29 | 403.00 | 414.18 | 3.98 | 6881.44 | 0 |
| 16 | 32 | 77.03 | 488.55 | 621.62 | 892.67 | 7.90 | 7719.44 | 0 |
| 16 | 48 | 77.69 | 673.50 | 1068.21 | 1092.37 | 10.90 | 7855.44 | 0 |
| 16 | 64 | 78.41 | 798.66 | 1397.64 | 1406.08 | 12.58 | 7473.44 | 0 |
| 16 | 128 | 78.27 | 1535.55 | 2267.65 | 2465.38 | 12.82 | 7159.44 | 0 |
| 32 | 16 | 74.66 | 213.75 | 402.88 | 414.34 | 3.98 | 6881.44 | 0 |
| 32 | 32 | 76.97 | 489.02 | 622.56 | 892.98 | 7.90 | 7719.44 | 0 |
| 32 | 48 | 77.80 | 673.02 | 1067.08 | 1088.14 | 10.87 | 7855.44 | 0 |
| 32 | 64 | 78.24 | 815.50 | 1399.71 | 1421.83 | 12.64 | 7473.44 | 0 |
| 32 | 128 | 78.28 | 1748.64 | 2436.67 | 2701.10 | 16.41 | 7223.44 | 0 |
| 48 | 16 | 74.76 | 212.30 | 402.79 | 413.86 | 3.98 | 6881.44 | 0 |
| 48 | 32 | 76.90 | 489.37 | 623.59 | 894.55 | 7.90 | 7719.44 | 0 |
| 48 | 48 | 77.80 | 673.11 | 1067.86 | 1088.15 | 10.87 | 7855.44 | 0 |
| 48 | 64 | 78.17 | 817.00 | 1399.95 | 1423.75 | 12.64 | 7473.44 | 0 |
| 48 | 128 | 78.22 | 1735.67 | 2436.48 | 2702.49 | 16.41 | 7223.44 | 0 |

A batch cap is a maximum, not a guaranteed realized batch. Concurrency below that cap cannot fill it. Peak GPU memory is whole-device NVML sampling every 50 ms, including warm-up; brief peaks can be missed. Queue delay is only the batch collection window; total waiting can exceed it under load.
