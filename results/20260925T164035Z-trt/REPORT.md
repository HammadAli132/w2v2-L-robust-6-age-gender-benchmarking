# Single-instance benchmark

Closed-loop local scheduler benchmark. Latency includes queueing, normalization, padding and GPU transfers; excludes HTTP and audio decoding. RPS is completed audio requests per second. Repeats are summarized by medians; raw request latencies are retained.

Highest RPS: batch cap 16, concurrency 128: 215.14 RPS, p95 827.82 ms.

Lowest p95: batch cap 32, concurrency 16: 201.76 RPS, p95 149.89 ms.

| Batch cap | Concurrency | RPS | p50 ms | p95 ms | p99 ms | Actual mean batch | Peak GPU MiB | Failed repeats |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 16 | 200.64 | 80.39 | 150.45 | 154.28 | 4.00 | 3207.44 | 0 |
| 8 | 32 | 213.36 | 148.39 | 326.07 | 330.52 | 7.94 | 3207.44 | 0 |
| 8 | 48 | 213.10 | 224.56 | 394.05 | 404.08 | 7.98 | 3207.44 | 0 |
| 8 | 64 | 214.72 | 290.50 | 417.14 | 446.66 | 7.95 | 3207.44 | 0 |
| 8 | 128 | 213.74 | 594.65 | 774.11 | 779.24 | 7.94 | 3207.44 | 0 |
| 16 | 16 | 200.59 | 80.33 | 150.29 | 154.35 | 3.99 | 3207.44 | 0 |
| 16 | 32 | 209.09 | 181.90 | 231.23 | 331.65 | 7.96 | 3207.44 | 0 |
| 16 | 48 | 210.55 | 251.08 | 394.70 | 401.52 | 11.03 | 3207.44 | 0 |
| 16 | 64 | 212.71 | 300.98 | 514.73 | 521.48 | 12.90 | 3207.44 | 0 |
| 16 | 128 | 215.14 | 565.24 | 827.82 | 901.40 | 12.95 | 3207.44 | 0 |
| 32 | 16 | 201.76 | 80.09 | 149.89 | 153.58 | 4.00 | 3207.44 | 0 |
| 32 | 32 | 210.63 | 181.13 | 228.64 | 329.63 | 7.94 | 3207.44 | 0 |
| 32 | 48 | 210.41 | 251.24 | 394.79 | 403.47 | 11.03 | 3207.44 | 0 |
| 32 | 64 | 211.47 | 304.70 | 518.49 | 525.38 | 12.92 | 3207.44 | 0 |
| 32 | 128 | 214.54 | 655.27 | 894.06 | 989.73 | 16.89 | 3207.44 | 0 |
| 48 | 16 | 200.25 | 80.47 | 150.49 | 154.37 | 3.99 | 3207.44 | 0 |
| 48 | 32 | 210.92 | 181.03 | 228.11 | 329.15 | 7.96 | 3207.44 | 0 |
| 48 | 48 | 210.41 | 251.04 | 394.98 | 402.95 | 11.04 | 3207.44 | 0 |
| 48 | 64 | 214.28 | 301.64 | 511.05 | 518.78 | 12.93 | 7273.44 | 0 |
| 48 | 128 | 212.77 | 658.89 | 900.11 | 994.70 | 16.90 | 3207.44 | 0 |

A batch cap is a maximum, not a guaranteed realized batch. Concurrency below that cap cannot fill it. Peak GPU memory is whole-device NVML sampling every 50 ms, including warm-up; brief peaks can be missed. Queue delay is only the batch collection window; total waiting can exceed it under load.
