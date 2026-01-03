"""
Step 2A: Run ONNX inference on ONE real saved sample.

Inputs:
- sitting_standing_model.onnx
- one_sample_150x20.npy  (shape: (150,20) or (1,150,20), dtype float32 preferred)

What it does:
- Loads the .npy sample
- Ensures shape is (1,150,20)
- Runs ONNX Runtime (CPUExecutionProvider)
- Prints input/output shapes and the output vector
"""

import argparse
import numpy as np
import onnxruntime as ort


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="sitting_standing_model.onnx", help="Path to ONNX model")
    ap.add_argument("--x_npy", default="one_sample_150x20.npy", help="Path to one sample .npy")
    args = ap.parse_args()

    # Load sample
    x = np.load(args.x_npy)

    # Ensure shape (1,150,20)
    if x.ndim == 2:
        x = x[None, :, :]
    if x.shape[1:] != (150, 20):
        raise ValueError(f"Expected (150,20) or (1,150,20). Got {x.shape}")

    # Ensure float32
    x = x.astype(np.float32)

    # ORT session
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name

    # Run inference
    y = sess.run(None, {inp_name: x})[0]  # first output
    y = np.asarray(y)

    print("✅ Step 2A: ONNX inference on one real sample")
    print(f"ONNX: {args.onnx}")
    print(f"NPY : {args.x_npy}")
    print(f"Input name : {inp_name}")
    print(f"Input shape: {x.shape}, dtype={x.dtype}")
    print(f"Output shape: {y.shape}, dtype={y.dtype}")
    print("Output vector:", y)

    pred = int(np.argmax(y, axis=-1)[0])
    print("Predicted class:", pred, "(0/1)")


if __name__ == "__main__":
    main()
