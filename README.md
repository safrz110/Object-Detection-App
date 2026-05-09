#  Real-Time Object Detection Web Application

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF6B35?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**An interactive web application for detecting 80+ object categories in images and video streams using state-of-the-art YOLOv8 architecture.**

[ Live Demo](#) · [ Documentation](#installation) · [ Report Bug](../../issues) · [ Request Feature](../../issues)

</div>

---

##  Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Variants](#-model-variants)
- [COCO Classes](#-coco-classes)
- [Deployment](#-deployment-streamlit-cloud)
- [Applications](#-real-world-applications)
- [Contributing](#-contributing)

---

##  Overview

This project implements a **production-ready object detection pipeline** using the YOLOv8 (You Only Look Once v8) neural network. The application exposes a polished **Streamlit web interface** that allows users to:

- **Upload images** and receive instant annotated results with bounding boxes and confidence scores
- **Process video files** frame-by-frame with real-time FPS measurement
- **Stream live webcam** feeds with on-device inference (local mode)
- **Tune detection parameters** (confidence, IoU, max detections) interactively

All inference runs on **standard CPU hardware** — no GPU required, making it accessible on any machine or cloud instance.

---

##  Features

| Feature | Details |
|---|---|
|  YOLOv8 Inference | Nano → Large model selection |
|  80 COCO Categories | Person, vehicle, animal, household objects, and more |
|  Confidence Threshold | Adjustable from 10% to 95% via sidebar slider |
|  NMS / IoU Control | Fine-tune Non-Maximum Suppression overlap threshold |
|  Video Processing | Upload MP4/AVI/MOV with per-frame annotation |
|  Live Webcam | Real-time feed detection (local deployment) |
|  Download Results | Export annotated images as PNG |
|  Detection Table | Full breakdown with labels, confidence, and bbox coords |
|  FPS Monitoring | Live frames-per-second overlay on video feeds |
|  Dark UI | GitHub-inspired dark theme with custom CSS |

---

##  Tech Stack

```
┌─────────────────────────────────────────────────────────┐
│                    Web Interface                        │
│                   Streamlit 1.32+                       │
├─────────────────────────────────────────────────────────┤
│               Detection Engine                          │
│         YOLOv8 (Ultralytics) via PyTorch               │
├──────────────────────┬──────────────────────────────────┤
│   Image Processing   │      Visualization               │
│     OpenCV 4.9+      │      Pillow / NumPy             │
├──────────────────────┴──────────────────────────────────┤
│                   Data / Analytics                      │
│                  Pandas · NumPy                         │
└─────────────────────────────────────────────────────────┘
```

---

##  Project Structure

```
object-detection-app/
│
├── app.py                    # Main Streamlit application entry point
│
├── utils/
│   ├── __init__.py           # Package exports
│   └── detection.py          # ObjectDetector class + drawing utilities
│
├── .streamlit/
│   └── config.toml           # Theme & server settings
│
├── requirements.txt          # Python dependencies
├── packages.txt              # System-level apt packages (Streamlit Cloud)
├── .gitignore
└── README.md
```

---

##  Installation

### Prerequisites

- Python 3.10 or higher
- pip / conda

### Step 1 — Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/object-detection-app.git
cd object-detection-app
```

### Step 2 — Create a Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# Or using conda
conda create -n yolo-app python=3.10
conda activate yolo-app
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** PyTorch (~2 GB) and Ultralytics (~50 MB) will be downloaded. YOLOv8 model weights (~6 MB for nano) are downloaded automatically on first run.

### Step 4 — Run the App

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

##  Usage

### Image Detection

1. Launch the app (`streamlit run app.py`)
2. Select a YOLOv8 model and adjust confidence from the **sidebar**
3. Go to ** Image Detection** tab
4. Upload a `.jpg`, `.png`, or `.webp` image
5. View the annotated result, detection table, and download the output

### Video Detection

1. Go to the **🎥 Video Detection** tab
2. Upload an MP4/AVI/MOV file
3. Set **max frames** and **frame skip** controls
4. Click ** Run Detection on Video**
5. Watch detections stream frame by frame

### Live Webcam (Local Only)

1. Run the app locally (not on Streamlit Cloud)
2. Select **Live Webcam** in the video tab
3. Click ** Start Webcam Feed**
4. Detections appear with real-time FPS overlay

### Sidebar Controls

| Control | Description |
|---|---|
|  Model | n / s / m / l — trade speed for accuracy |
|  Confidence | Filter out low-confidence detections |
|  IoU (NMS) | Control duplicate-box suppression |
|  Max Detections | Cap objects per frame |
|  Labels / Conf display | Toggle text overlays |
|  Box Thickness | 1–5 px bounding box weight |

---

##  Model Variants

| Model | Parameters | mAP50-95 | CPU Speed | Best For |
|---|---|---|---|---|
| **YOLOv8n** ⚡ | 3.2M | 37.3 | ~1 ms | Real-time, edge devices |
| **YOLOv8s** | 11.2M | 44.9 | ~1.2 ms | Balanced speed/accuracy |
| **YOLOv8m** | 25.9M | 50.2 | ~1.8 ms | Higher accuracy |
| **YOLOv8l** | 43.7M | 52.9 | ~2.4 ms | Production accuracy |
| **YOLOv8x** | 68.2M | 53.9 | ~3.5 ms | Maximum accuracy |

> *Speeds measured on NVIDIA A100. CPU speeds will be slower but all variants work without GPU.*

---

##  COCO Classes

The model detects all **80 MS-COCO categories**:

<details>
<summary>Click to expand all 80 classes</summary>

```
person          bicycle         car             motorcycle      airplane
bus             train           truck           boat            traffic light
fire hydrant    stop sign       parking meter   bench           bird
cat             dog             horse           sheep           cow
elephant        bear            zebra           giraffe         backpack
umbrella        handbag         tie             suitcase        frisbee
skis            snowboard       sports ball     kite            baseball bat
baseball glove  skateboard      surfboard       tennis racket   bottle
wine glass      cup             fork            knife           spoon
bowl            banana          apple           sandwich        orange
broccoli        carrot          hot dog         pizza           donut
cake            chair           couch           potted plant    bed
dining table    toilet          tv              laptop          mouse
remote          keyboard        cell phone      microwave       oven
toaster         sink            refrigerator    book            clock
vase            scissors        teddy bear      hair drier      toothbrush
```

</details>

---

##  Deployment — Streamlit Cloud

### One-Click Deploy

1. Push this repo to **GitHub**
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repo → `app.py` as the main file
5. Click **Deploy** — done in ~2 minutes!

### Manual Streamlit Cloud Setup

```
Repository:   github.com/YOUR_USERNAME/object-detection-app
Branch:       main
Main file:    app.py
Python:       3.10
```

> **packages.txt** handles system-level OpenCV dependencies (libgl1, libglib2, ffmpeg) automatically on Streamlit Cloud.

### Other Platforms

<details>
<summary>Docker</summary>

```dockerfile
FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 ffmpeg
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t yolo-detector .
docker run -p 8501:8501 yolo-detector
```

</details>

<details>
<summary>Hugging Face Spaces</summary>

Create a new **Streamlit** Space and push the repo. Add a `README.md` header:

```yaml
---
title: YOLOv8 Object Detector
emoji: 
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
---
```

</details>

---

##  Real-World Applications

### 🔧 Manufacturing Quality Control
Real-time defect detection on assembly lines. Identify misaligned components, surface scratches, or foreign objects at production speed without halting the line.

###  Security Surveillance
Multi-camera person and vehicle tracking. Trigger alerts on unauthorized object detection (bags, weapons) in restricted zones.

###  Autonomous Systems
Core perception module for self-driving vehicles — pedestrian detection, traffic sign recognition, and obstacle avoidance.

###  Medical Imaging
With fine-tuned weights, detect tumors, lesions, and anomalies in radiological scans. The pipeline supports custom datasets via Ultralytics training API.

###  Retail Analytics
Customer foot-traffic counting, shelf occupancy monitoring, and queue detection for store optimization.

###  Agriculture / Drone Imagery
Crop disease detection, pest identification, and automated yield counting from UAV footage.

---

##  Roadmap

- [ ] Custom model training UI (upload dataset → train → deploy)
- [ ] Multi-camera feed support
- [ ] Object tracking across frames (ByteTrack / BoT-SORT)
- [ ] REST API endpoint (`/detect`) using FastAPI
- [ ] Export to ONNX / TensorRT for GPU acceleration
- [ ] Batch image processing with ZIP download
- [ ] Heatmap visualization for detection density

---

##  Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

##  License

Distributed under the MIT License. See `LICENSE` for more information.

---

##  Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — state-of-the-art detection architecture
- [Streamlit](https://streamlit.io) — rapid ML app framework
- [OpenCV](https://opencv.org) — computer vision primitives
- [MS COCO Dataset](https://cocodataset.org) — 80-class training benchmark

---

<div align="center">

**Built with  using YOLOv8 + Streamlit**

Author
Sarfaraz Ali
