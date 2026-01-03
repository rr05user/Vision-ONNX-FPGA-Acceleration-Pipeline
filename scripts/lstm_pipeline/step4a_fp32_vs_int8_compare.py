import numpy as np
import onnxruntime as ort

FP32_MODEL = "sitting_standing_model.onnx"
INT8_MODEL = "quant_out/sitting_standing_model_int8.onnx"
INPUT_NPY = "one_sample_150x20.npy"

# Load input
x = np.load(INPUT_NPY).astype(np.float32)
if x.ndim == 2:
    x = x[None, :, :]  # (1,150,20)

# FP32 inference
sess_fp32 = ort.InferenceSession(FP32_MODEL, providers=["CPUExecutionProvider"])
inp_name = sess_fp32.get_inputs()[0].name
y_fp32 = sess_fp32.run(None, {inp_name: x})[0]

# INT8 inference
sess_int8 = ort.InferenceSession(INT8_MODEL, providers=["CPUExecutionProvider"])
y_int8 = sess_int8.run(None, {inp_name: x})[0]

# Compare
abs_diff = np.abs(y_fp32 - y_int8)

print("FP32 output:", y_fp32)
print("INT8 output:", y_int8)
print("Abs diff:", abs_diff)
print("Max abs diff:", abs_diff.max())
print("Mean abs diff:", abs_diff.mean())
