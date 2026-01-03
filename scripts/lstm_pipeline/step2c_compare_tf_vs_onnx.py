"""
Step 2C: Compare TensorFlow (SavedModel) vs ONNX Runtime outputs on the same input.

Requires:
- model_temp/  (SavedModel directory produced by your training/checkpoint)
- sitting_standing_model.onnx
- one_sample_150x20.npy  OR a batch file like calib_156_samples.npy

What it does:
- Loads X from .npy (supports (150,20), (1,150,20), or (N,150,20))
- Runs TF model: tf.keras.models.load_model(savedmodel_dir)
- Runs ONNX model: onnxruntime
- Prints:
  - output shapes
  - max abs diff, mean abs diff
  - prints first few rows for inspection
"""

import argparse
import numpy as np
import onnxruntime as ort
import tensorflow as tf


def load_x(path: str) -> np.ndarray:
    x = np.load(path)
    if x.ndim == 2:
        x = x[None, :, :]  # (150,20) -> (1,150,20)
    if x.ndim != 3 or x.shape[1:] != (150, 20):
        raise ValueError(f"Expected (150,20) or (N,150,20). Got {x.shape}")
    return x.astype(np.float32)


def run_tf(savedmodel_dir: str, x: np.ndarray) -> np.ndarray:
    model = tf.keras.models.load_model(savedmodel_dir)
    y = model(x, training=False).numpy()
    return y.astype(np.float32)


def run_onnx(onnx_path: str, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    y = sess.run(None, {inp_name: x})[0]
    return np.asarray(y).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--savedmodel", default="model_temp", help="SavedModel directory (e.g., model_temp)")
    ap.add_argument("--onnx", default="sitting_standing_model.onnx", help="Path to ONNX model")
    ap.add_argument("--x_npy", default="one_sample_150x20.npy", help="Input .npy: (150,20) or (N,150,20)")
    ap.add_argument("--n", type=int, default=1, help="Use first N samples from x_npy (useful if batch file)")
    args = ap.parse_args()

    x = load_x(args.x_npy)
    x = x[: min(args.n, len(x))]

    print("✅ Step 2C: TF vs ONNX comparison")
    print("SavedModel:", args.savedmodel)
    print("ONNX     :", args.onnx)
    print("X NPY    :", args.x_npy)
    print("X shape  :", x.shape, x.dtype)

    y_tf = run_tf(args.savedmodel, x)
    y_ox = run_onnx(args.onnx, x)

    if y_tf.shape != y_ox.shape:
        raise ValueError(f"Shape mismatch: TF {y_tf.shape} vs ONNX {y_ox.shape}")

    diff = np.abs(y_tf - y_ox)
    print("\nOutput shape:", y_tf.shape)
    print("Max abs diff :", float(diff.max()))
    print("Mean abs diff:", float(diff.mean()))

    # Show a few rows
    show = min(5, len(x))
    print("\nFirst rows (TF):")
    print(y_tf[:show])
    print("\nFirst rows (ONNX):")
    print(y_ox[:show])
    print("\nFirst rows (abs diff):")
    print(diff[:show])

    # Simple pass/fail suggestion
    # Softmax outputs usually match within small tolerance (1e-3 to 1e-2)
    tol = 1e-2
    print(f"\nSuggested tolerance check (<= {tol}):", "PASS ✅" if diff.max() <= tol else "CHECK ⚠️")


if __name__ == "__main__":
    main()
