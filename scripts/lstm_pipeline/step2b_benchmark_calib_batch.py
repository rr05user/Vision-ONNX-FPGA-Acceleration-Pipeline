"""
Step 2B: Benchmark ONNX inference on a REAL calibration batch (CPU baseline).

Default inputs:
- ONNX:  sitting_standing_model.onnx
- CALIB: calib_156_samples.npy   (shape: (N,150,20))

What it does:
- Loads calibration batch
- Runs warmup
- Times multiple runs
- Prints mean/p50/p90/p99 latency
- Two modes:
  - batch: run all N samples at once
  - loop: run one sample at a time (streaming-like)
"""

import argparse
import time
import numpy as np
import onnxruntime as ort


def run_once(sess, inp_name, X):
    return sess.run(None, {inp_name: X})[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="sitting_standing_model.onnx", help="Path to ONNX model")
    ap.add_argument("--calib_npy", default="calib_156_samples.npy", help="Path to calibration .npy (N,150,20)")
    ap.add_argument("--n", type=int, default=50, help="How many samples to benchmark from the file")
    ap.add_argument("--runs", type=int, default=20, help="Timed runs")
    ap.add_argument("--warmup", type=int, default=5, help="Warmup runs (not timed)")
    ap.add_argument("--mode", choices=["batch", "loop"], default="batch",
                    help="batch: run all N samples at once. loop: run one sample at a time.")
    args = ap.parse_args()

    # Load calibration data
    X = np.load(args.calib_npy)

    # Validate shape
    if X.ndim != 3 or X.shape[1:] != (150, 20):
        raise ValueError(f"Expected (N,150,20). Got {X.shape} from {args.calib_npy}")

    # Force float32 (matches ONNX input type)
    X = X.astype(np.float32)

    # Limit to n samples
    X = X[: min(args.n, len(X))]

    # ORT session
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name

    print("✅ Step 2B: CPU benchmark (real data)")
    print(f"ONNX : {args.onnx}")
    print(f"CALIB: {args.calib_npy}")
    print(f"Mode : {args.mode}")
    print(f"X    : {X.shape} dtype={X.dtype}")
    print(f"Input name: {inp_name}")

    # Warmup
    for _ in range(args.warmup):
        if args.mode == "batch":
            _ = run_once(sess, inp_name, X)
        else:
            for i in range(len(X)):
                _ = run_once(sess, inp_name, X[i:i+1])

    # Timed runs
    times_ms = []
    y = None

    for _ in range(args.runs):
        t0 = time.perf_counter()

        if args.mode == "batch":
            y = run_once(sess, inp_name, X)
        else:
            ys = []
            for i in range(len(X)):
                ys.append(run_once(sess, inp_name, X[i:i+1]))
            y = np.concatenate(ys, axis=0)

        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    times_ms = np.array(times_ms, dtype=np.float64)

    print("\nOutput shape:", np.asarray(y).shape)
    print("\nLatency over", args.runs, "runs:")
    print(f"  mean: {times_ms.mean():.3f} ms")
    print(f"  p50 : {np.percentile(times_ms, 50):.3f} ms")
    print(f"  p90 : {np.percentile(times_ms, 90):.3f} ms")
    print(f"  p99 : {np.percentile(times_ms, 99):.3f} ms")

    print(f"\nApprox per-sample (mean/N): {(times_ms.mean()/len(X)):.3f} ms/sample")


if __name__ == "__main__":
    main()
