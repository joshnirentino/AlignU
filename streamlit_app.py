# streamlit_app.py
"""
Streamlit wrapper for posture_detector_updated.py
"""

import io
import numpy as np
import streamlit as st
from PIL import Image
import cv2
import posture_detector_updated as pd

st.set_page_config(page_title="Ergonomics AI — Posture Detector", layout="centered")

st.title("🧍‍♀️ Ergonomics AI — Posture Detector")
st.write("Upload an image to analyze your posture using MediaPipe.")

mp_status = "Available" if pd.HAS_MEDIAPIPE else "Not Available — using fallback"
st.info(f"**MediaPipe status:** {mp_status}")

uploaded = st.file_uploader("Upload an image (jpg, png, jpeg)", type=["jpg", "jpeg", "png"])


# ----------------------------------------------------------------
# Convert uploaded file to OpenCV BGR image
# ----------------------------------------------------------------
def uploaded_file_to_cv2(file):
    bytes_data = file.read()
    np_arr = np.frombuffer(bytes_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img


# ----------------------------------------------------------------
# PROCESS IMAGE THROUGH YOUR ORIGINAL CODE
# ----------------------------------------------------------------
def process_frame(frame_bgr):
    h, w = frame_bgr.shape[:2]

    features = None
    points = None

    # MediaPipe first
    if pd.HAS_MEDIAPIPE:
        try:
            mp_pose = pd.mp.solutions.pose
            with mp_pose.Pose(static_image_mode=True) as pose:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)
                if getattr(res, "pose_landmarks", None):
                    out = pd.features_mediapipe(res.pose_landmarks.landmark, w, h)
                    if out:
                        features, points = out
                else:
                    # fallback to face mode
                    out = pd.features_face(frame_bgr)
                    if out:
                        features, points = out
        except Exception as e:
            st.warning(f"MediaPipe error: {e}. Using face fallback.")
            out = pd.features_face(frame_bgr)
            if out:
                features, points = out
    else:
        out = pd.features_face(frame_bgr)
        if out:
            features, points = out

    # Evaluate posture
    checks = pd.evaluate_ergonomics(features, points, frame_bgr)

    # Annotate
    try:
        annotated = pd.annotate_frame_with_meters(frame_bgr, features, points, "", checks)
    except Exception:
        annotated = pd.annotate_frame(frame_bgr, features, points, "")

    # Scoring
    scores = pd.score_from_features(features)
    mood_text, _ = pd.mood_from_scores(scores)

    return annotated, checks, scores, mood_text


# ----------------------------------------------------------------
# STREAMLIT UI
# ----------------------------------------------------------------
if uploaded:
    with st.spinner("Processing..."):
        frame = uploaded_file_to_cv2(uploaded)

        if frame is None:
            st.error("Could not read image file.")
        else:
            annotated, checks, scores, mood_text = process_frame(frame)

            # DEBUG
            st.subheader("DEBUG — Internal values")
            st.json(checks)
            st.json(scores)
            st.write("Mood:", mood_text)

            # Display annotated image
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(rgb, caption="Posture Analysis Result", use_column_width=True)

            # Metrics
            st.subheader("📊 Scores")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Neck", scores.get("neck", 0))
            col2.metric("Back", scores.get("back", 0))
            col3.metric("Head", scores.get("head", 0))
            col4.metric("Overall", scores.get("overall10", 0))

            # Feedback
            st.subheader("💬 Feedback")
            st.write(f"**{mood_text}**")

            # Suggestions (DEDUPED)
            st.subheader("📝 Suggestions")
            sugg = []

            for key, (ok, msg) in checks.items():
                if ok is False and key in pd.TIPS:
                    sugg.append(pd.TIPS[key])

            # remove duplicates while keeping order
            sugg = list(dict.fromkeys(sugg))

            if sugg:
                for s in sugg:
                    st.markdown("- " + s)
            else:
                st.write("Your posture looks good!")

            # Download annotated image
            buf = io.BytesIO()
            Image.fromarray(rgb).save(buf, format="PNG")
            st.download_button("Download Annotated Image", buf.getvalue(), "annotated.png")

else:
    st.info("Upload an image above to begin.")
