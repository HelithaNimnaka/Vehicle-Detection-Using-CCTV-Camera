# Vehicle Detection Using CCTV Camera 🚗📹

A high-performance vehicle detection and counting system built using YOLOv11 and the UA-DETRAC dataset. This project focuses on real-time surveillance applications using CCTV video feeds.

---

## 🎞️ Demo

![Vehicle Tracking Demo](Vehicle-Detection-Using-CCTV-Camera/Vehicle%20Tracking.gif)

---
## 🚀 Overview

- **Model**: YOLOv11m (Ultralytics)
- **Dataset**: UA-DETRAC (real-world traffic footage)
- **Vehicle Classes**: Car, Truck, Van, Bus
- **Performance**:
  - Up to **346 FPS** with TensorRT INT8
  - Cross-platform deployment: PyTorch, ONNX, TensorRT, OpenVINO
  - Multithreaded tracking support

## 📦 Dataset

- Source: [UA-DETRAC](https://www.kaggle.com/datasets/dtrnngc/ua-detrac-dataset)
- Training images: 83,791  
- Validation/Test: 28,170  
- Image Size: 640x640  
- Classes: `car`, `truck`, `van`, `bus`

## 🧠 Model Architecture

- **Backbone**: Enhanced convolutional layers with C3k/C3k2 modules
- **Neck**: PANet structure with SPPF block
- **Head**: Multi-scale detection with DFL loss
- **Optimizer**: SGD with early stopping
- **Augmentations**: Mosaic, MixUp, HSV shifts, flipping, perspective

## 🛠️ Inference Backends

| Backend     | FPS  | Precision | mAP@50 |
|-------------|------|-----------|--------|
| PyTorch     | 85   | 0.718     | 0.712  |
| ONNX (GPU)  | 65   | 0.721     | 0.711  |
| TensorRT    | 346  | 0.684     | 0.653  |
| OpenVINO    | 18   | 0.709     | 0.696  |

## 🧪 Real-Time Testing

- Integrated with **ByteTrack** for multi-object tracking
- Supports **frame-by-frame** and **batch** tracking modes
- **Vehicle Count Output** per video stream

## 🧰 Hardware Used

- GPU: NVIDIA RTX 2080 Ti (11GB)
- CPU: Intel Core i9-9900K (8C/16T)
- RAM: 32 GB DDR4

## 📊 Evaluation Metrics

- **Overall mAP@50–95**: 0.507 (PyTorch), 0.486 (ONNX), 0.448 (TensorRT INT8)
- **Vehicle counts** match ground truth in most test scenarios

## 🔄 Export Formats

- `.pt` (PyTorch)
- `.onnx` (ONNX Runtime)
- `.engine` (TensorRT INT8)
- OpenVINO IR (INT8)

## 📈 Future Improvements

- Expand to more classes or general vehicle detection
- Fine-tune on multi-weather and nighttime datasets
- Optimize for deployment-specific hardware

## 📎 References

- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
- [YOLOv11 paper](https://arxiv.org/html/2410.17725v1)
- [OpenVINO Toolkit](https://docs.openvino.ai/2025/index.html)

---

### 👤 Author

**Helitha Nimnaka**  
GitHub: [HelithaNimnaka/Vehicle-Detection-Using-CCTV-Camera](https://github.com/HelithaNimnaka/Vehicle-Detection-Using-CCTV-Camera)
