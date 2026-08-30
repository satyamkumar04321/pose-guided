import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import sys
import glob

# Ensure current directory is in sys.path for cloud imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pose_detector import PoseDetector

# =====================================================================
# PAGE CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="PoseGuide - AI Body Shape Analyzer & Pose Recommender",
    page_icon="🧍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# CUSTOM CSS STYLING
# =====================================================================
st.markdown("""
<style>
    /* Dark Glassmorphism Styling */
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a855f7 0%, #6366f1 50%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.8rem;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #cbd5e1;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .status-badge {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
    .status-success {
        background-color: rgba(34, 197, 94, 0.2);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.4);
    }
    .status-warning {
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# INITIALIZE POSE DETECTOR CACHED
# =====================================================================
@st.cache_resource
def get_detector():
    return PoseDetector()


detector = get_detector()

# Pose Folders mapping
POSE_FOLDERS = {
    "Oval": "Poses/oval",
    "Triangle": "Poses/triangle",
    "Rectangle": "Poses/rectangle",
    "Trapezium": "Poses/trapezium",
    "Inverted Triangle": "Poses/inverted_triangle"
}

BODY_DESCRIPTIONS = {
    "Oval": "Shoulders are narrower than waist (~0.70 ratio). Recommended poses add structure to shoulders and create diagonal body lines.",
    "Triangle": "Hips/waist are wider than shoulders (~0.85 ratio). Recommended poses draw focus upwards and widen upper posture.",
    "Rectangle": "Shoulder and waist widths are nearly balanced (~1.00 ratio). Recommended poses introduce waist angles and dynamic twists.",
    "Trapezium": "Shoulders are slightly broader than waist (~1.12 ratio). Highly versatile athletic silhouette; works well with relaxed or structured framing.",
    "Inverted Triangle": "Shoulders are significantly broader than waist (~1.30 ratio). Poses balance lower body proportions with hip hand placement or angled stances."
}


# Helper function to load pose recommendation images
def load_pose_images(body_type):
    folder = POSE_FOLDERS.get(body_type)
    if not folder or not os.path.exists(folder):
        return []
    image_paths = sorted(glob.glob(os.path.join(folder, "*.jpg")))
    return image_paths


# Helper function to process image array through PoseDetector
def process_image(img_rgb):
    # Convert RGB PIL/Streamlit image to BGR for PoseDetector
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    result = detector.detect_pose(img_bgr)
    return result


# =====================================================================
# SIDEBAR NAVIGATION
# =====================================================================
with st.sidebar:
    st.image("https://img.icons8.com/isometric-folders/100/body-type.png", width=64)
    st.title("PoseGuide AI")
    st.caption("AI-Powered Body Shape & Modeling Pose Recommender")
    st.divider()

    mode = st.radio(
        "Select Mode:",
        ["📷 Live Browser Camera", "📤 Upload Image", "📖 Body Shape Guide"],
        index=0
    )

    st.divider()
    st.markdown("### 💡 Quick Tips")
    st.info("""
    - Stand **3-4 feet away** facing the camera.
    - Ensure your **full body** (shoulders to feet) is visible.
    - Wear contrasting clothing against your background for ideal silhouette segmentation.
    """)

    st.caption("Built with MediaPipe, OpenCV & Streamlit")

# =====================================================================
# MAIN HEADER
# =====================================================================
st.markdown('<div class="main-header">🧍 PoseGuide AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Real-time computer vision silhouette extraction, body shape classification & tailored pose recommendations</div>',
    unsafe_allow_html=True
)

# =====================================================================
# MODE 1: LIVE BROWSER CAMERA
# =====================================================================
if mode == "📷 Live Browser Camera":
    st.subheader("📷 Take a Photo for Analysis")
    st.caption("Use your laptop webcam or mobile camera to capture your full-body pose.")

    camera_image = st.camera_input("Capture Pose")

    if camera_image is not None:
        pil_img = Image.open(camera_image).convert("RGB")
        img_np = np.array(pil_img)

        with st.spinner("Processing pose landmarks & silhouette extraction..."):
            result = process_image(img_np)

        annotated_rgb = cv2.cvtColor(result["frame"], cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.image(annotated_rgb, caption="Annotated Pose & Scanning Lines", use_container_width=True)

        with col2:
            if result["valid_pose"]:
                st.markdown('<div class="status-badge status-success">✅ Valid Pose Captured</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="status-badge status-warning">⚠️ {result["message"]}</div>', unsafe_allow_html=True)

            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Body Shape</div>
                    <div class="metric-value" style="color: #a855f7;">{result['body_type']}</div>
                </div>
                """, unsafe_allow_html=True)
            with m_col2:
                ratio_val = f"{result['body_ratio']:.2f}" if result['body_ratio'] else "N/A"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Shoulder/Waist Ratio</div>
                    <div class="metric-value">{ratio_val}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            m_col3, m_col4 = st.columns(2)
            with m_col3:
                s_w = f"{result['shoulder_width']} px" if result['shoulder_width'] else "N/A"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Shoulder Width</div>
                    <div class="metric-value" style="font-size: 1.4rem;">{s_w}</div>
                </div>
                """, unsafe_allow_html=True)
            with m_col4:
                w_w = f"{result['waist_width']} px" if result['waist_width'] else "N/A"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Waist Width</div>
                    <div class="metric-value" style="font-size: 1.4rem;">{w_w}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.write(f"**Classification Confidence:** `{result['confidence']}%`")

            # Download annotated image option
            buf_img = Image.fromarray(annotated_rgb)
            img_byte_arr = cv2.imencode('.jpg', result["frame"])[1].tobytes()
            st.download_button(
                label="📥 Download Annotated Image",
                data=img_byte_arr,
                file_name="pose_analysis.jpg",
                mime="image/jpeg"
            )

            # Segmentation Mask expander
            with st.expander("🔍 View Silhouette Segmentation Mask"):
                st.image(result["mask"], caption="Binary Selfie Segmentation Mask", use_container_width=True)

        st.divider()

        # =============================================================
        # POSE RECOMMENDATIONS GALLERY
        # =============================================================
        body_type = result["body_type"]
        if body_type != "Unknown":
            st.subheader(f"💃 Recommended Modeling Poses for {body_type} Shape")
            st.info(BODY_DESCRIPTIONS.get(body_type, ""))

            pose_images = load_pose_images(body_type)
            if pose_images:
                cols = st.columns(4)
                for idx, img_p in enumerate(pose_images[:12]):
                    with cols[idx % 4]:
                        rec_img = Image.open(img_p)
                        st.image(rec_img, caption=f"Pose #{idx + 1}", use_container_width=True)
            else:
                st.warning("No pose recommendations found for this body type.")

# =====================================================================
# MODE 2: UPLOAD IMAGE
# =====================================================================
elif mode == "📤 Upload Image":
    st.subheader("📤 Upload a Photo for Pose Analysis")
    st.caption("Upload a full-body picture (JPG, PNG) to analyze body shape and get pose recommendations.")

    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(pil_img)

        with st.spinner("Analyzing pose landmarks & silhouette..."):
            result = process_image(img_np)

        annotated_rgb = cv2.cvtColor(result["frame"], cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.image(annotated_rgb, caption="Annotated Pose & Scanning Lines", use_container_width=True)

        with col2:
            if result["valid_pose"]:
                st.markdown('<div class="status-badge status-success">✅ Valid Pose Detected</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="status-badge status-warning">⚠️ {result["message"]}</div>', unsafe_allow_html=True)

            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Body Shape</div>
                    <div class="metric-value" style="color: #a855f7;">{result['body_type']}</div>
                </div>
                """, unsafe_allow_html=True)
            with m_col2:
                ratio_val = f"{result['body_ratio']:.2f}" if result['body_ratio'] else "N/A"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Shoulder/Waist Ratio</div>
                    <div class="metric-value">{ratio_val}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            m_col3, m_col4 = st.columns(2)
            with m_col3:
                s_w = f"{result['shoulder_width']} px" if result['shoulder_width'] else "N/A"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Shoulder Width</div>
                    <div class="metric-value" style="font-size: 1.4rem;">{s_w}</div>
                </div>
                """, unsafe_allow_html=True)
            with m_col4:
                w_w = f"{result['waist_width']} px" if result['waist_width'] else "N/A"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Waist Width</div>
                    <div class="metric-value" style="font-size: 1.4rem;">{w_w}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.write(f"**Classification Confidence:** `{result['confidence']}%`")

            # Segmentation Mask expander
            with st.expander("🔍 View Silhouette Segmentation Mask"):
                st.image(result["mask"], caption="Binary Selfie Segmentation Mask", use_container_width=True)

        st.divider()

        # Pose recommendations
        selected_shape = st.selectbox(
            "Explore Recommendations for Body Shape:",
            list(POSE_FOLDERS.keys()),
            index=list(POSE_FOLDERS.keys()).index(result["body_type"]) if result["body_type"] in POSE_FOLDERS else 0
        )

        st.info(BODY_DESCRIPTIONS.get(selected_shape, ""))

        pose_images = load_pose_images(selected_shape)
        if pose_images:
            cols = st.columns(4)
            for idx, img_p in enumerate(pose_images[:12]):
                with cols[idx % 4]:
                    rec_img = Image.open(img_p)
                    st.image(rec_img, caption=f"Pose #{idx + 1}", use_container_width=True)
        else:
            st.warning("No pose images available for this body type.")

# =====================================================================
# MODE 3: BODY SHAPE GUIDE
# =====================================================================
else:
    st.subheader("📖 Body Shape Classification Reference Guide")
    st.write(
        "PoseGuide uses empirical shoulder-to-waist width ratios measured at key anatomical landmarks from selfie segmentation masks."
    )

    st.markdown("""
    | Body Shape | Ideal Ratio (Shoulder / Waist) | Description & Styling Advice |
    | :--- | :---: | :--- |
    | 🟢 **Oval** | `0.70` | Waist width is wider than shoulder width. Poses with jacket layering or angled shoulders enhance structure. |
    | 🟡 **Triangle** | `0.85` | Hips/waist are wider than shoulders. Poses that broaden upper posture or place hands high add balance. |
    | 🔵 **Rectangle** | `1.00` | Shoulders and waist are balanced. Poses with hand-on-hip, torso twists, and bent knees create dynamic curves. |
    | 🟣 **Trapezium** | `1.12` | Broad shoulders with a balanced taper to the waist. Versatile shape suited for straight and relaxed postures. |
    | 🔴 **Inverted Triangle** | `1.30` | Shoulders are significantly wider than waist. Poses focusing on hip accentuation or lower-body angles create harmony. |
    """)

    st.divider()
    st.subheader("✨ Browse All Pose Recommendation Decks")

    guide_shape = st.selectbox("Select Body Shape Deck:", list(POSE_FOLDERS.keys()))
    st.write(BODY_DESCRIPTIONS.get(guide_shape, ""))

    pose_imgs = load_pose_images(guide_shape)
    if pose_imgs:
        cols = st.columns(4)
        for idx, img_p in enumerate(pose_imgs[:16]):
            with cols[idx % 4]:
                rec_img = Image.open(img_p)
                st.image(rec_img, caption=f"{guide_shape} Pose #{idx + 1}", use_container_width=True)
