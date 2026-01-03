from ultralytics import YOLO

# Pick a small model for FPGA-friendly deployment first
# (later you can swap to pose model if you want keypoints directly)
MODEL = "yolov8n.pt"   # or your own trained weights .pt
OUT = "yolo_fp32.onnx"

m = YOLO(MODEL)

# Export to ONNX
# opset 13 is generally safe; simplify True helps compilers
m.export(format="onnx", opset=13, simplify=True, dynamic=False)

print("✅ Exported YOLO to ONNX (check your runs/export folder).")
