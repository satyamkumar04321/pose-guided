# 🧍 PoseGuide: AI-Powered Body Shape Analyzer & Pose Recommender

PoseGuide is a real-time computer vision and machine learning application that detects a user's pose, extracts their physical silhouette, classifies their body shape, and recommends personalized modeling and fashion poses tailored specifically to their body type.

By leveraging **MediaPipe Pose**, **MediaPipe Selfie Segmentation**, **OpenCV**, and **Streamlit**, the application automates pose validation, body ratio calculation, and style recommendation directly from a live web camera or uploaded images.

---

## 🚀 Key Features

* **🌐 Web & Cloud Ready**: Fully deployable web application built with **Streamlit** (supports phone & desktop browsers).
* **📷 Live Camera & Upload Modes**: Take photos live in browser (`st.camera_input`) or upload photos (JPG, PNG).
* **⏱️ Anatomical Landmark & Pose Validation**: Ensures full body visibility, standing alignment, and stability.
* **👤 Silhouette-Based Width Extraction**: Uses selfie segmentation to generate a precise binary mask and scans horizontal lines at key anatomical heights to determine exact physical width.
* **📊 Accurate Body Ratio Calculation**: Automatically calculates the ratio between shoulder width and waist width.
* **🧠 Body Shape Classification**: Uses empirical mathematical profiles to classify the user's body into one of five standard body shapes:
  * 🟢 **Oval** (Ideal Ratio ~`0.70`)
  * 🟡 **Triangle** (Ideal Ratio ~`0.85`)
  * 🔵 **Rectangle** (Ideal Ratio ~`1.00`)
  * 🟣 **Trapezium** (Ideal Ratio ~`1.12`)
  * 🔴 **Inverted Triangle** (Ideal Ratio ~`1.30`)
* **💃 Tailored Pose Recommendations**: Dynamically loads and displays professional fashion/modeling poses matching the user's classified body shape.

---

## 🛠️ Tech Stack

* **Language**: Python 3.8+
* **Web Framework**: Streamlit
* **Computer Vision & Tracking**: 
  * OpenCV (`opencv-python-headless`)
  * [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose.html)
  * [MediaPipe Selfie Segmentation](https://google.github.io/mediapipe/solutions/selfie_segmentation.html)
* **Data Processing & Imaging**: NumPy, Pillow, Math, Glob

---

## 📁 Repository Structure

```filepath
├── .streamlit/
│   └── config.toml            # Streamlit custom theme configuration
├── Poses/                     # Recommended pose datasets (images)
│   ├── inverted_triangle/
│   ├── oval/
│   ├── rectangle/
│   ├── trapezium/
│   └── triangle/
├── src/
│   └── pose_detector.py       # Core PoseDetector engine (Landmarks, Silhouette & Ratios)
├── app.py                     # Streamlit Web Application (Cloud Deployable)
├── main.py                    # Local OpenCV Desktop Webcam Script
├── packages.txt               # Linux system dependencies for cloud hosting
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation (this file)
```

---

## 🚀 Quick Start & Local Execution

### 1. Clone the repository & install dependencies

```bash
git clone https://github.com/satyamkumar04321/pose-guided.git
cd pose-guided

# (Optional) Create virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Web Application (Recommended)

Launch the Streamlit web dashboard in your browser:
```bash
streamlit run app.py
```

### 3. Run the Desktop OpenCV App (Local only)

For offline desktop webcam processing:
```bash
python main.py
```

---

## ☁️ Deployment Guide

### Option 1: Streamlit Community Cloud (Recommended - Free & 1-Click)

1. Push your repository to **GitHub**.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Click **"New App"**.
4. Select your repository, branch (`main`), and set Main file path to `app.py`.
5. Click **"Deploy!"**. Streamlit Cloud will automatically detect `packages.txt` (for Linux OpenCV GL libraries) and `requirements.txt`.

### Option 2: Hugging Face Spaces

1. Create a new Space on Hugging Face and select **Streamlit** as the SDK.
2. Push your files (including `app.py`, `requirements.txt`, `packages.txt`, and `Poses/`).
3. Space will build and launch automatically.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
