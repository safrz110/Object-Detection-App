"""
Real-Time Object Detection Web Application
Tech: Python, YOLOv8, OpenCV, Streamlit, PIL
Author: Built with YOLOv8 + Streamlit
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import tempfile
import os
from utils.detection import ObjectDetector, draw_detections, get_class_colors

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="YOLOv8 Object Detector",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main theme */
    .main { background-color: #0d1117; }
    .stApp { background-color: #0d1117; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
        border-right: 1px solid #30363d;
    }

    /* Metric cards */
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        margin: 4px 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #58a6ff;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Detection badge */
    .detection-badge {
        display: inline-block;
        background: #1f6feb;
        color: white;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        margin: 2px;
        font-weight: 600;
    }

    /* Header */
    .app-header {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 24px 32px;
        margin-bottom: 24px;
    }

    /* Info boxes */
    .info-box {
        background: #161b22;
        border-left: 4px solid #58a6ff;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 8px 0;
        color: #c9d1d9;
        font-size: 0.9rem;
    }

    /* Upload area styling */
    [data-testid="stFileUploader"] {
        background: #161b22;
        border: 2px dashed #30363d;
        border-radius: 8px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #238636, #2ea043);
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        width: 100%;
        padding: 8px 16px;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2ea043, #3fb950);
    }

    /* Slider */
    .stSlider > div > div > div { background: #238636 !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        background: #161b22;
        border: 1px solid #30363d;
        color: #c9d1d9;
        border-radius: 6px 6px 0 0;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: #1f6feb !important;
        color: white !important;
        border-color: #1f6feb !important;
    }

    /* Divider */
    hr { border-color: #30363d !important; }

    h1, h2, h3 { color: #f0f6fc !important; }
    p, li { color: #c9d1d9 !important; }
    label { color: #c9d1d9 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Initialize Session State
# ─────────────────────────────────────────────
if "detector" not in st.session_state:
    st.session_state.detector = None
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "total_detections" not in st.session_state:
    st.session_state.total_detections = 0
if "frames_processed" not in st.session_state:
    st.session_state.frames_processed = 0


# ─────────────────────────────────────────────
# Sidebar — Controls
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    # Model selection
    model_size = st.selectbox(
        "🤖 YOLOv8 Model",
        options=["yolov8n", "yolov8s", "yolov8m", "yolov8l"],
        index=0,
        help="n=Nano (fastest), s=Small, m=Medium, l=Large (most accurate)",
    )

    # Confidence threshold
    confidence = st.slider(
        "🎯 Confidence Threshold",
        min_value=0.10,
        max_value=0.95,
        value=0.45,
        step=0.05,
        help="Minimum confidence score to display a detection",
    )

    # IOU threshold
    iou_thresh = st.slider(
        "📐 IoU Threshold (NMS)",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.05,
        help="Non-Maximum Suppression overlap threshold",
    )

    # Max detections
    max_det = st.number_input(
        "🔢 Max Detections per Frame",
        min_value=1,
        max_value=300,
        value=50,
        step=10,
    )

    st.markdown("---")

    # Display options
    st.markdown("### 🎨 Display Options")
    show_labels = st.checkbox("Show Labels", value=True)
    show_confidence = st.checkbox("Show Confidence %", value=True)
    show_fps = st.checkbox("Show FPS Overlay", value=True)
    bbox_thickness = st.slider("Bounding Box Thickness", 1, 5, 2)

    st.markdown("---")

    # Load model button
    if st.button("🚀 Load / Reload Model", use_container_width=True):
        with st.spinner(f"Loading {model_size}..."):
            try:
                st.session_state.detector = ObjectDetector(
                    model_name=model_size,
                    conf_threshold=confidence,
                    iou_threshold=iou_thresh,
                    max_det=max_det,
                )
                st.session_state.model_loaded = True
                st.success(f"✅ {model_size} loaded!")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    # Model status indicator
    if st.session_state.model_loaded:
        st.markdown(
            '<div class="info-box">🟢 <b>Model Active</b> — Ready to detect</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="info-box">🔴 <b>No Model</b> — Click Load above</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 📊 Session Stats")
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value">{st.session_state.total_detections}</div>'
        f'<div class="metric-label">Total Detections</div>'
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value">{st.session_state.frames_processed}</div>'
        f'<div class="metric-label">Frames Processed</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# Main Content
# ─────────────────────────────────────────────
st.markdown(
    """
    <div class="app-header">
        <h1 style="margin:0; font-size:2rem;">🎯 Real-Time Object Detection</h1>
        <p style="margin:4px 0 0 0; color:#8b949e;">
            YOLOv8 · OpenCV · Streamlit &nbsp;|&nbsp; 80+ COCO Object Categories
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Auto-load nano model on first run
if not st.session_state.model_loaded:
    with st.spinner("🔄 Auto-loading YOLOv8n (nano)..."):
        try:
            st.session_state.detector = ObjectDetector(
                model_name="yolov8n",
                conf_threshold=confidence,
                iou_threshold=iou_thresh,
                max_det=max_det,
            )
            st.session_state.model_loaded = True
        except Exception as e:
            st.warning(f"Auto-load failed: {e}. Please load manually from sidebar.")

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📷 Image Detection", "🎥 Video Detection", "ℹ️ Model Info"])


# ══════════════════════════════════════════════
# TAB 1 — IMAGE DETECTION
# ══════════════════════════════════════════════
with tab1:
    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("### 📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Drag & drop or browse",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            label_visibility="collapsed",
        )

        if uploaded_file:
            pil_image = Image.open(uploaded_file).convert("RGB")
            st.image(pil_image, caption="Original Image", use_container_width=True)

            col_w, col_h = st.columns(2)
            col_w.metric("Width", f"{pil_image.width}px")
            col_h.metric("Height", f"{pil_image.height}px")

    with col_result:
        st.markdown("### 🔍 Detection Result")

        if uploaded_file and st.session_state.model_loaded:
            with st.spinner("Running inference..."):
                # Update thresholds from sidebar
                st.session_state.detector.update_params(confidence, iou_thresh, max_det)

                img_np = np.array(pil_image)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                t0 = time.time()
                detections = st.session_state.detector.detect(img_bgr)
                inference_time = (time.time() - t0) * 1000  # ms

                # Draw boxes
                result_img = draw_detections(
                    img_bgr.copy(),
                    detections,
                    show_labels=show_labels,
                    show_conf=show_confidence,
                    thickness=bbox_thickness,
                )
                result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

            st.image(result_rgb, caption="Detections", use_container_width=True)

            # Update stats
            st.session_state.total_detections += len(detections)
            st.session_state.frames_processed += 1

            # Metrics row
            m1, m2, m3 = st.columns(3)
            m1.metric("Objects Found", len(detections))
            m2.metric("Inference", f"{inference_time:.1f} ms")
            m3.metric("Est. FPS", f"{1000/inference_time:.1f}" if inference_time > 0 else "N/A")

            # Detection breakdown
            if detections:
                st.markdown("#### 📋 Detected Objects")
                class_counts: dict[str, int] = {}
                for det in detections:
                    label = det["label"]
                    class_counts[label] = class_counts.get(label, 0) + 1

                badges = " ".join(
                    f'<span class="detection-badge">{cls} ×{cnt}</span>'
                    for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1])
                )
                st.markdown(badges, unsafe_allow_html=True)

                # Detailed table
                with st.expander("📊 Full Detection Table"):
                    import pandas as pd
                    df = pd.DataFrame(detections)
                    df["confidence"] = df["confidence"].map("{:.1%}".format)
                    df.columns = ["Label", "Confidence", "X1", "Y1", "X2", "Y2"]
                    st.dataframe(df, use_container_width=True, hide_index=True)

                # Download result
                result_pil = Image.fromarray(result_rgb)
                from io import BytesIO
                buf = BytesIO()
                result_pil.save(buf, format="PNG")
                st.download_button(
                    "⬇️ Download Result",
                    data=buf.getvalue(),
                    file_name="detection_result.png",
                    mime="image/png",
                )
        elif not st.session_state.model_loaded:
            st.info("⚙️ Load the model from the sidebar first.")
        else:
            st.info("⬆️ Upload an image to begin detection.")


# ══════════════════════════════════════════════
# TAB 2 — VIDEO DETECTION
# ══════════════════════════════════════════════
with tab2:
    st.markdown("### 🎥 Video / Webcam Detection")

    video_source = st.radio(
        "Input Source",
        ["Upload Video File", "Live Webcam"],
        horizontal=True,
    )

    if video_source == "Upload Video File":
        video_file = st.file_uploader(
            "Upload a video",
            type=["mp4", "avi", "mov", "mkv"],
            label_visibility="collapsed",
        )

        if video_file and st.session_state.model_loaded:
            # Save to temp file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(video_file.read())
            tfile.flush()

            col_ctrl1, col_ctrl2 = st.columns([2, 1])
            max_frames = col_ctrl1.slider("Max Frames to Process", 10, 500, 100, 10)
            frame_skip = col_ctrl2.number_input("Frame Skip", 1, 10, 2)

            if st.button("▶️ Run Detection on Video"):
                cap = cv2.VideoCapture(tfile.name)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps_video = cap.get(cv2.CAP_PROP_FPS) or 30

                st.info(f"📹 Video: {total_frames} frames @ {fps_video:.0f} FPS")

                stframe = st.empty()
                progress = st.progress(0)
                fps_display = st.empty()

                frame_idx = 0
                processed = 0
                all_fps = []

                st.session_state.detector.update_params(confidence, iou_thresh, max_det)

                while cap.isOpened() and processed < max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_idx += 1
                    if frame_idx % frame_skip != 0:
                        continue

                    t0 = time.time()
                    detections = st.session_state.detector.detect(frame)
                    elapsed = time.time() - t0
                    fps_val = 1.0 / elapsed if elapsed > 0 else 0
                    all_fps.append(fps_val)

                    result = draw_detections(
                        frame.copy(),
                        detections,
                        show_labels=show_labels,
                        show_conf=show_confidence,
                        thickness=bbox_thickness,
                    )

                    if show_fps:
                        cv2.putText(
                            result,
                            f"FPS: {fps_val:.1f}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 255, 0),
                            2,
                        )

                    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                    stframe.image(result_rgb, use_container_width=True)

                    processed += 1
                    progress.progress(processed / max_frames)
                    fps_display.markdown(
                        f"**Frame {processed}/{max_frames}** | "
                        f"FPS: `{fps_val:.1f}` | "
                        f"Objects: `{len(detections)}`"
                    )

                    st.session_state.total_detections += len(detections)
                    st.session_state.frames_processed += 1

                cap.release()
                os.unlink(tfile.name)

                avg_fps = np.mean(all_fps) if all_fps else 0
                st.success(
                    f"✅ Done! Processed {processed} frames | "
                    f"Avg FPS: {avg_fps:.1f}"
                )

    else:  # Live Webcam
        st.markdown(
            '<div class="info-box">📡 <b>Live Webcam</b> requires running locally '
            "(not supported on Streamlit Cloud). "
            "Clone the repo and run <code>streamlit run app.py</code>.</div>",
            unsafe_allow_html=True,
        )

        if st.session_state.model_loaded:
            run_webcam = st.button("📷 Start Webcam Feed")
            stop_webcam = st.button("⏹️ Stop")

            if run_webcam:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("❌ Could not open webcam. Check camera permissions.")
                else:
                    stframe = st.empty()
                    fps_display = st.empty()
                    st.session_state.detector.update_params(confidence, iou_thresh, max_det)

                    frame_count = 0
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        t0 = time.time()
                        detections = st.session_state.detector.detect(frame)
                        elapsed = time.time() - t0
                        fps_val = 1.0 / elapsed if elapsed > 0 else 0

                        result = draw_detections(
                            frame.copy(),
                            detections,
                            show_labels=show_labels,
                            show_conf=show_confidence,
                            thickness=bbox_thickness,
                        )

                        if show_fps:
                            cv2.putText(
                                result,
                                f"FPS: {fps_val:.1f} | Objects: {len(detections)}",
                                (10, 35),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.0,
                                (0, 255, 100),
                                2,
                            )

                        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                        stframe.image(result_rgb, channels="RGB", use_container_width=True)
                        fps_display.markdown(f"⚡ FPS: `{fps_val:.1f}` | 🎯 Objects: `{len(detections)}`")

                        st.session_state.total_detections += len(detections)
                        st.session_state.frames_processed += 1
                        frame_count += 1

                    cap.release()


# ══════════════════════════════════════════════
# TAB 3 — MODEL INFO
# ══════════════════════════════════════════════
with tab3:
    st.markdown("### 🤖 YOLOv8 Architecture Overview")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Model Variants")
        import pandas as pd
        model_df = pd.DataFrame({
            "Model": ["YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l", "YOLOv8x"],
            "Params (M)": [3.2, 11.2, 25.9, 43.7, 68.2],
            "mAP50-95": [37.3, 44.9, 50.2, 52.9, 53.9],
            "Speed (ms)": [0.99, 1.20, 1.83, 2.39, 3.53],
        })
        st.dataframe(model_df, use_container_width=True, hide_index=True)

    with col_b:
        st.markdown("#### COCO Categories (sample)")
        categories = [
            "person", "bicycle", "car", "motorcycle", "airplane",
            "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee",
        ]
        badges = " ".join(
            f'<span class="detection-badge">{c}</span>' for c in categories
        )
        st.markdown(badges + " <i>...and 50 more</i>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🏭 Real-World Applications")
    apps = {
        "🔧 Manufacturing QC": "Detect defects, misaligned parts, and foreign objects on assembly lines in real-time.",
        "🔐 Security Surveillance": "Track people, vehicles, and suspicious objects across multiple camera feeds.",
        "🚗 Autonomous Systems": "Enable self-driving vehicles to perceive pedestrians, signs, and obstacles.",
        "🏥 Medical Imaging": "Identify anomalies in X-rays, MRIs, and pathology slides.",
        "🛒 Retail Analytics": "Count customers, track shelf occupancy, and analyze foot traffic patterns.",
        "🌾 Agriculture": "Monitor crop health, detect pests, and count produce yield from drone imagery.",
    }
    for app, desc in apps.items():
        st.markdown(
            f'<div class="info-box"><b>{app}</b><br>{desc}</div>',
            unsafe_allow_html=True,
        )

    if st.session_state.model_loaded and st.session_state.detector:
        st.markdown("---")
        st.markdown("#### 🔬 Loaded Model Details")
        info = st.session_state.detector.get_model_info()
        for k, v in info.items():
            st.markdown(f"**{k}:** `{v}`")
