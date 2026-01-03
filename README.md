# Vision ONNX FPGA Acceleration Pipeline

An end-to-end **computer vision hardware acceleration pipeline** demonstrating how deep learning models move from training to deployment on FPGA hardware using **ONNX, INT8 quantization, and Xilinx Vitis AI**.

This project mirrors real industry workflows used in **AI hardware acceleration, edge AI, and FPGA-based inference systems**.

---

## 🔍 Project Overview

The pipeline integrates **two AI workloads**:

1. **Temporal Classification (LSTM)**  
   - Classifies human posture (e.g., sitting vs standing)
   - Runs efficiently on CPU using ONNX Runtime (INT8)

2. **Spatial Vision Front-End (YOLOv8)**  
   - Performs real-time object detection
   - Compiled into a **DPU-ready `.xmodel`** for FPGA execution using **Vitis AI**

The final architecture reflects a **heterogeneous deployment**:
- **FPGA (DPU)** → heavy convolutional vision workload  
- **CPU** → sequential / control-heavy LSTM inference

---

## 🧠 Model Details

### 1️⃣ LSTM Temporal Classifier
- **Framework**: TensorFlow / Keras
- **Input shape**: `(N, 150, 20)`
  - 150-frame temporal window
  - 20 pose-based features per frame
- **Output shape**: `(N, 2)` (binary classification)
- **Export path**:
  - Keras → ONNX → ONNX Runtime
- **INT8 quantization** validated against FP32

### 2️⃣ YOLOv8 Vision Model
- **Framework**: Ultralytics YOLOv8
- **Input shape**: `(1, 3, 640, 640)`
- **Output shape**: `(1, 84, 8400)`
- **Export path**:
  - PyTorch → ONNX → INT8 (QDQ) → Vitis AI `.xmodel`

---

## ⚙️ Pipeline Stages

### Step 1 — Train & Export (LSTM)
- Train LSTM model in TensorFlow
- Export to ONNX
- Verify graph correctness

### Step 2 — ONNX Runtime Validation
- Run inference on real inputs
- Benchmark FP32 vs INT8
- Compare ONNX outputs vs TensorFlow

**Numerical accuracy**
- Max absolute diff: ~0.0096
- Mean absolute diff: ~0.0096

### Step 3 — Quantization (INT8)
- Static INT8 quantization using calibration samples
- QDQ format for hardware compatibility

### Step 4 — YOLO Export & Analysis
- Export YOLOv8 to ONNX
- Inspect operator graph for FPGA compatibility
- Confirm CNN-only ops (Conv, Mul, Add, Concat)

### Step 5 — Vitis AI Compilation
- Compile YOLO INT8 ONNX to `.xmodel`
- Target platform: **KV260 (Zynq UltraScale+)**
- Tool: `vai_c_xir`

---

## ⏱️ Performance Results (CPU)

### LSTM ONNX Runtime Benchmark
| Mode | Mean Latency | Per-sample |
|----|----|----|
| FP32 | 238.9 ms | 4.78 ms |
| INT8 | 273.1 ms | 5.46 ms |

> INT8 on CPU may not be faster; the benefit is realized on FPGA/DPU hardware.

---
## 📁 Repository Structure




