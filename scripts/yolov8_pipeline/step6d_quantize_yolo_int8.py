
import os
import glob
import numpy as np
import cv2
import onnx
from onnxruntime.quantization import (
    quantize_static, CalibrationDataReader,
    QuantType, QuantFormat, CalibrationMethod
)

FP32_MODEL = "yolov8n.onnx"
INT8_MODEL = "yolov8n_int8_qdq.onnx"
IMG_DIR = "calib_images"
IMG_SIZE = 640
BATCH = 1   # keep 1 for YOLO calibration unless you really know batching

def preprocess(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to read {img_path}")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img[:, :, ::-1]  # BGR->RGB
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC->CHW
    img = np.expand_dims(img, 0)        # BCHW
    return img

class ImageFolderReader(CalibrationDataReader):
    def __init__(self, folder):
        exts = ["*.jpg","*.jpeg","*.png","*.bmp","*.webp"]
        self.files = []
        for e in exts:
            self.files += glob.glob(os.path.join(folder, e))
        self.files = sorted(self.files)
        if len(self.files) == 0:
            raise FileNotFoundError(f"No images found in {folder}")
        self.i = 0

    def get_next(self):
        if self.i >= len(self.files):
            return None
        x = preprocess(self.files[self.i])
        self.i += 1
        return {"images": x}

def main():
    print("Quantizing:", FP32_MODEL)
    print("Calib dir :", IMG_DIR)

    # Validate input model
    m = onnx.load(FP32_MODEL)
    onnx.checker.check_model(m)
    print("✓ FP32 ONNX validated")

    dr = ImageFolderReader(IMG_DIR)

    quantize_static(
        model_input=FP32_MODEL,
        model_output=INT8_MODEL,
        calibration_data_reader=dr,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=True,
        reduce_range=False,
        optimize_model=False
    )

    qm = onnx.load(INT8_MODEL)
    onnx.checker.check_model(qm)
    print("✓ INT8 ONNX validated")
    print("✅ Wrote:", INT8_MODEL)

if __name__ == "__main__":
    main()

