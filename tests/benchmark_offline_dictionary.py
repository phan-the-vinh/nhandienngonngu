"""Empirical benchmark for OfflineDictionary.

Measures:
- Dictionary load time
- File size
- Word count
- Average search latency
- P95 search latency
- RAM memory consumption via tracemalloc
"""
from __future__ import annotations

import json
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.suggestion.dictionary import OfflineDictionary

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "sample_dictionary.txt"


def run_benchmark(iterations: int = 2000) -> dict[str, object]:
    file_size_bytes = FIXTURE_PATH.stat().st_size

    # 1. Measure load time & memory footprint
    tracemalloc.start()
    t0 = time.perf_counter()
    d = OfflineDictionary.load(FIXTURE_PATH)
    load_time_seconds = time.perf_counter() - t0
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    word_count = len(d)

    # 2. Measure search latency across diverse prefixes
    test_prefixes = [
        "A", "AP", "APP", "B", "BO", "C", "CA", "D", "E", "F",
        "G", "H", "HEL", "HELLO", "I", "K", "L", "M", "N", "O",
        "P", "Q", "R", "S", "T", "W", "Y", "Z", "XYZNOTEXIST", ""
    ]

    latencies_us: list[float] = []

    # Warmup
    for p in test_prefixes:
        d.search(p, limit=5)

    # Measurement loop
    for _ in range(iterations // len(test_prefixes)):
        for p in test_prefixes:
            t_start = time.perf_counter()
            d.search(p, limit=5)
            t_end = time.perf_counter()
            latencies_us.append((t_end - t_start) * 1_000_000)

    avg_latency_us = statistics.mean(latencies_us)
    p50_latency_us = statistics.median(latencies_us)
    latencies_sorted = sorted(latencies_us)
    p95_idx = int(0.95 * len(latencies_sorted))
    p95_latency_us = latencies_sorted[p95_idx]
    max_latency_us = max(latencies_us)
    min_latency_us = min(latencies_us)

    results = {
        "file_path": str(FIXTURE_PATH),
        "file_size_bytes": file_size_bytes,
        "word_count": word_count,
        "load_time_ms": round(load_time_seconds * 1000, 3),
        "peak_memory_bytes": peak_mem,
        "peak_memory_kb": round(peak_mem / 1024, 2),
        "total_queries_measured": len(latencies_us),
        "avg_latency_us": round(avg_latency_us, 3),
        "p50_latency_us": round(p50_latency_us, 3),
        "p95_latency_us": round(p95_latency_us, 3),
        "min_latency_us": round(min_latency_us, 3),
        "max_latency_us": round(max_latency_us, 3),
    }

    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    run_benchmark()
