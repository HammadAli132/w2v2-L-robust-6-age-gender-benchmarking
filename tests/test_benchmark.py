import asyncio
import time
import unittest
from unittest.mock import patch

import numpy as np

from bench.benchmark import trial
from bench.prepare import bucket_samples, pack


class CountingRuntime:
    def __init__(self):
        self.calls = []
        self.bucket_sets = []
        self.active = False

    def infer(self, signal, lengths):
        if self.active:
            raise AssertionError("More than one simultaneous model execution")
        self.active = True
        self.calls.append(len(lengths))
        self.bucket_sets.append({bucket_samples(int(length)) for length in lengths})
        time.sleep(0.001)
        self.active = False
        return np.zeros((len(lengths), 1)), np.zeros((len(lengths), 3))


class BenchmarkTests(unittest.TestCase):
    def test_padding_does_not_change_normalization(self):
        a = np.array([1, 2, 3, 4], dtype=np.float32)
        b = np.arange(12, dtype=np.float32)
        batch, lengths = pack([a, b])
        alone, _ = pack([a])
        np.testing.assert_array_equal(batch[0, :4], alone[0])
        np.testing.assert_array_equal(batch[0, 4:], 0)
        np.testing.assert_array_equal(lengths, [4, 12])

    def test_single_worker_and_request_accounting(self):
        runtime = CountingRuntime()
        with patch("torch.cuda.synchronize"), patch("torch.cuda.reset_peak_memory_stats"), \
             patch("torch.cuda.max_memory_allocated", return_value=0), \
             patch("torch.cuda.max_memory_reserved", return_value=0):
            result = asyncio.run(trial(runtime, [np.ones(16000, dtype=np.float32)],
                                       batch_size=8, concurrency=16, duration=0.05,
                                       delay_ms=0, warmup=0))
        self.assertGreater(result["requests"], 0)
        self.assertEqual(sum(runtime.calls), result["requests"])
        self.assertLessEqual(max(runtime.calls), 8)
        self.assertEqual(len(result["raw_requests"]), result["requests"])
        self.assertTrue(all(row[2] >= row[3] >= 0 for row in result["raw_requests"]))
        self.assertAlmostEqual(result["rps"], result["requests"] / result["elapsed_s"])

    def test_lengths_are_bucketed_and_batch_is_bounded_by_concurrency(self):
        runtime = CountingRuntime()
        clips = [np.ones(n, dtype=np.float32) for n in (12000, 15000, 40000, 47000)]
        with patch("torch.cuda.synchronize"), patch("torch.cuda.reset_peak_memory_stats"), \
             patch("torch.cuda.max_memory_allocated", return_value=0), \
             patch("torch.cuda.max_memory_reserved", return_value=0):
            result = asyncio.run(trial(runtime, clips, batch_size=48, concurrency=16,
                                       duration=0.05, delay_ms=0, warmup=0))
        self.assertLessEqual(result["actual_batch_max"], 16)
        self.assertTrue(all(len(buckets) == 1 for buckets in runtime.bucket_sets))
        self.assertEqual(sum(result["bucket_counts"].values()), result["requests"])

    def test_execution_failure_does_not_hang_clients(self):
        runtime = CountingRuntime()
        runtime.infer = lambda *_: (_ for _ in ()).throw(RuntimeError("simulated OOM"))
        with patch("torch.cuda.synchronize"), patch("torch.cuda.reset_peak_memory_stats"):
            with self.assertRaisesRegex(RuntimeError, "simulated OOM"):
                asyncio.run(asyncio.wait_for(trial(runtime, [np.ones(16000, dtype=np.float32)],
                    batch_size=8, concurrency=16, duration=0.05, delay_ms=0, warmup=0), timeout=1))


if __name__ == "__main__":
    unittest.main()
