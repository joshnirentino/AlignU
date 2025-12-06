"""
Ergonomics AI — Posture Detection for Uploaded Image
File: posture_detector_updated.py
Language: Python 3.10+

This file:
- Scores posture on a 0-10 scale (neck, trunk/back, head alignment) and computes an overall 0-10.
- Uses MediaPipe pose when available; falls back to a Haar face detector for minimal features.
- Shows two windows side-by-side: an OpenCV image window (left) and a Tkinter feedback popup (right).
- Avoids drawing large text on the image; the popup contains friendly, de-duplicated tips.


"""

import argparse
import math
import os
import sys
import warnings
from collections import deque

# Silence noisy protobuf deprecation warnings which are harmless here
warnings.filterwarnings("ignore", category=UserWarning, module=r"google\.protobuf")

SMOOTHING_WINDOW = 5
STATIC_POSTURE_THRESHOLD_S = 4.0  # seconds (for video static posture warnings)

# -----------------------------
# Safe imports
# -----------------------------
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except Exception:
    mp = None
    HAS_MEDIAPIPE = False

try:
    import numpy as np
except Exception:
    np = None

try:
    import cv2
except Exception:
    print("ERROR: OpenCV (cv2) is required. Install with: python -m pip install opencv-python")
    sys.exit(1)

# -----------------------------
# Ergonomic thresholds (ISO-11226-based approximations)
# -----------------------------
THRESHOLDS = {
    'neck_flexion': {'good_max': 30.0, 'warning_max': 50.0},  # degrees
    'trunk_sagittal': {'good_max': 20.0, 'warning_max': 60.0},
    'knee_sitting': {'good_min': 110.0, 'good_max': 135.0},
}

# -----------------------------
# Utilities
# -----------------------------

def clamp(v, a, b):
    return max(a, min(b, v))


def angle_between_points(a, b, c):
    ax, ay = a
    bx, by = b
    cx, cy = c
    ba = (ax - bx, ay - by)
    bc = (cx - bx, cy - by)
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    norma = math.hypot(ba[0], ba[1])
    normb = math.hypot(bc[0], bc[1])
    if norma == 0 or normb == 0:
        return 0.0
    cosang = dot / (norma * normb)
    cosang = max(-1.0, min(1.0, cosang))
    return math.degrees(math.acos(cosang))


def landmark_xy(lm, w, h):
    return int(lm.x * w), int(lm.y * h)

# -----------------------------
# Scoring helper (0-10)
# -----------------------------

def classify_flexion_angle(value, good_max, warning_max):
    """Return (category, score_0_10).

    Mapping:
      good -> 7..10
      warning -> 4..6
      bad -> 0..3
    """
    if value <= good_max:
        frac = value / (good_max + 1e-6)
        score = 7 + int((1.0 - frac) * 3)
        return 'good', clamp(score, 7, 10)
    if value <= warning_max:
        frac = (value - good_max) / (warning_max - good_max)
        score = 6 - int(frac * 2)
        return 'warning', clamp(score, 4, 6)
    frac = min((value - warning_max) / (warning_max if warning_max > 0 else 1.0), 1.0)
    score = 3 - int(frac * 3)
    return 'bad', clamp(score, 0, 3)

# -----------------------------
# Feature extraction
# -----------------------------

def features_mediapipe(landmarks, w, h):
    try:
        ls = landmark_xy(landmarks[11], w, h)  # left shoulder
        rs = landmark_xy(landmarks[12], w, h)  # right shoulder
        le = landmark_xy(landmarks[13], w, h)  # left elbow
        re = landmark_xy(landmarks[14], w, h)  # right elbow
        lw = landmark_xy(landmarks[15], w, h)  # left wrist
        rw = landmark_xy(landmarks[16], w, h)  # right wrist
        lh = landmark_xy(landmarks[23], w, h)  # left hip
        rh = landmark_xy(landmarks[24], w, h)  # right hip
        lk = landmark_xy(landmarks[25], w, h)  # left knee
        rk = landmark_xy(landmarks[26], w, h)  # right knee
        la = landmark_xy(landmarks[27], w, h)  # left ankle
        ra = landmark_xy(landmarks[28], w, h)  # right ankle
        nose = landmark_xy(landmarks[0], w, h)
    except Exception:
        return None

    mid_sh = ((ls[0] + rs[0]) // 2, (ls[1] + rs[1]) // 2)
    mid_hp = ((lh[0] + rh[0]) // 2, (lh[1] + rh[1]) // 2)

    # neck flexion: angle between vector (shoulder_mid -> nose) and vertical up
    v_nose = (nose[0] - mid_sh[0], nose[1] - mid_sh[1])
    mag_v = math.hypot(v_nose[0], v_nose[1])
    if mag_v == 0:
        neck_flexion_deg = 0.0
    else:
        cosang = max(-1.0, min(1.0, (-v_nose[1]) / mag_v))
        neck_flexion_deg = math.degrees(math.acos(cosang))

    # trunk sagittal: shoulders->hips vector vs vertical
    v_sh_hp = (mid_hp[0] - mid_sh[0], mid_hp[1] - mid_sh[1])
    mag_v2 = math.hypot(v_sh_hp[0], v_sh_hp[1])
    if mag_v2 == 0:
        trunk_sagittal_deg = 0.0
    else:
        cosang2 = max(-1.0, min(1.0, v_sh_hp[1] / mag_v2))
        trunk_sagittal_deg = math.degrees(math.acos(cosang2))

    left_elbow_angle = angle_between_points(ls, le, lw)
    right_elbow_angle = angle_between_points(rs, re, rw)
    left_knee_angle = angle_between_points(lh, lk, la)
    right_knee_angle = angle_between_points(rh, rk, ra)

    features = {
        'neck_flexion_deg': neck_flexion_deg,
        'trunk_sagittal_deg': trunk_sagittal_deg,
        'left_elbow_angle': left_elbow_angle,
        'right_elbow_angle': right_elbow_angle,
        'left_knee_angle': left_knee_angle,
        'right_knee_angle': right_knee_angle,
    }
    points = {'mid_sh': mid_sh, 'ls': ls, 'rs': rs, 'lh': lh, 'rh': rh, 'nose': nose, 'lk': lk, 'rk': rk, 'la': la, 'ra': ra}
    return features, points

# Fallback face-based features
FACE_CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(FACE_CASCADE) if os.path.exists(FACE_CASCADE) else None

def features_face(frame):
    if face_cascade is None:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    fh, fw = frame.shape[:2]
    head_y = (y + h / 2) / fh
    size = (w * h) / (fw * fh)
    neck = 180.0 * (1.0 - head_y)
    back = min(180.0, 180.0 * size * 5.0)
    features = {'neck_flexion_deg': neck, 'trunk_sagittal_deg': back}
    points = {'face_box': (x, y, w, h)}
    return features, points

# -----------------------------
# Ergonomic checks & tips
# -----------------------------

def evaluate_ergonomics(features, points, frame=None):
    checks = {}
    if features is None:
        checks['overall'] = (False, 'No features detected')
        return checks

    nf = features.get('neck_flexion_deg', 0.0)
    n_cat, n_score = classify_flexion_angle(nf, THRESHOLDS['neck_flexion']['good_max'], THRESHOLDS['neck_flexion']['warning_max'])
    checks['neck_flexion'] = (n_cat == 'good', f"Neck flexion={nf:.1f} deg ({n_cat}, score={n_score})")

    ts = features.get('trunk_sagittal_deg', 0.0)
    t_cat, t_score = classify_flexion_angle(ts, THRESHOLDS['trunk_sagittal']['good_max'], THRESHOLDS['trunk_sagittal']['warning_max'])
    checks['trunk_sagittal'] = (t_cat == 'good', f"Trunk sagittal tilt={ts:.1f} deg ({t_cat}, score={t_score})")

    # head alignment: horizontal offset of nose from shoulder midline (if available)
    head_dx = None
    if points is not None and 'nose' in points and 'mid_sh' in points:
        nose_x = points['nose'][0]; midx = points['mid_sh'][0]
        dx = abs(nose_x - midx)
        head_dx = dx
        if frame is not None:
            fw = frame.shape[1]
            thresh_px = max(20, 0.05 * fw)
        else:
            thresh_px = 40
        checks['head_aligned'] = (dx <= thresh_px, f"Head offset={dx:.1f}px (<= {thresh_px:.1f}px good)")
    else:
        checks['head_aligned'] = (None, 'Head alignment not measurable')

    le = features.get('left_elbow_angle', None)
    re = features.get('right_elbow_angle', None)
    if le is not None:
        checks['left_elbow_90'] = (70.0 <= le <= 110.0, f"Left elbow angle={le:.1f} (70-110 deg recommended)")
    if re is not None:
        checks['right_elbow_90'] = (70.0 <= re <= 110.0, f"Right elbow angle={re:.1f} (70-110 deg recommended)")

    lk = features.get('left_knee_angle', None)
    rk = features.get('right_knee_angle', None)
    if lk is not None:
        checks['left_knee_sitting'] = (THRESHOLDS['knee_sitting']['good_min'] <= lk <= THRESHOLDS['knee_sitting']['good_max'], f"Left knee angle={lk:.1f} (recommended 110-135 deg)")
    if rk is not None:
        checks['right_knee_sitting'] = (THRESHOLDS['knee_sitting']['good_min'] <= rk <= THRESHOLDS['knee_sitting']['good_max'], f"Right knee angle={rk:.1f} (recommended 110-135 deg)")

    checks['lumbar_support'] = (None, 'Lumbar support: place backrest support at lower back (manual check)')
    checks['viewing_angle'] = (None, 'Viewing angle: place work at 10-30deg below eye level (manual check)')

    measured = [v[0] for v in checks.values() if v[0] is not None]
    if len(measured) == 0:
        overall = False
    else:
        overall = sum(1 for x in measured if x) >= max(1, len(measured)//2)
    checks['overall'] = (overall, f"Overall ergonomics pass={overall}")

    if head_dx is not None:
        features['head_offset_px'] = head_dx
    else:
        features['head_offset_px'] = None

    return checks

# user-friendly tips mapping
TIPS = {
    'head_aligned': "Minor head offset detected - check that your head is centered over your shoulders.",
    'trunk_sagittal': "Forward trunk tilt detected - sit back into the chair and support the lumbar region.",
    'neck_flexion': "Slight forward head posture detected — gently bring head back toward neutral. If you still feel discomfort, adjust monitor height.",
    'left_elbow_90': "Keep elbows at ~90 degrees and close to the body.",
    'right_elbow_90': "Keep elbows at ~90 degrees and close to the body.",
    'left_knee_sitting': "Keep knees between 110-135 degrees when seated; adjust seat or use a footrest.",
    'right_knee_sitting': "Keep knees between 110-135 degrees when seated; adjust seat or use a footrest.",
}

EXTRA_TIPS = [
    "Your torso is leaning - pull the chair closer to avoid reaching forward.",
    "Uneven shoulders found - relax both sides and let them settle evenly.",
    "Side-bending detected - keep both feet grounded to stay centered.",
]

# -----------------------------
# Scoring (0-10) and annotation helpers
# -----------------------------

def score_from_features(features):
    if features is None:
        return {'neck':0, 'back':0, 'head':0, 'overall10':0}
    neck_deg = features.get('neck_flexion_deg', 0.0)
    neck_score = classify_flexion_angle(neck_deg, THRESHOLDS['neck_flexion']['good_max'], THRESHOLDS['neck_flexion']['warning_max'])[1]

    back_deg = features.get('trunk_sagittal_deg', 0.0)
    back_score = classify_flexion_angle(back_deg, THRESHOLDS['trunk_sagittal']['good_max'], THRESHOLDS['trunk_sagittal']['warning_max'])[1]

    head_offset = features.get('head_offset_px', None)
    if head_offset is None:
        head_score = 10
    else:
        thresh = 40.0
        if head_offset <= thresh:
            head_score = 10
        else:
            head_score = clamp(int(10 - ((head_offset - thresh) / (3 * thresh - thresh)) * 10), 0, 10)

    overall = int(round((neck_score + back_score + head_score) / 3.0))
    return {'neck': neck_score, 'back': back_score, 'head': head_score, 'overall10': overall}


def annotate_frame(frame, features, points, label):
    out = frame.copy()
    if points is not None:
        for k, v in points.items():
            if isinstance(v, tuple) and len(v) == 2:
                cv2.circle(out, v, 3, (255,140,0), -1)
    return out


def annotate_frame_with_meters(frame, features, points, label, checks):
    # keep image clean — only small keypoints
    return annotate_frame(frame, features, points, label)

# -----------------------------
# Small helpers re: report & UI
# -----------------------------

def write_report(report_path, checks):
    with open(report_path, 'w', encoding='utf-8') as f:
        for k, (ok, msg) in checks.items():
            f.write(f"{k}: {ok} - {msg}\n")


def mood_from_scores(scores):
    avg = scores.get('overall10', 0)
    if avg >= 9:
        return ("Optimal Alignment", (0,180,0))
    if avg >= 7:
        return ("Slight Misalignment", (60,160,180))
    if avg >= 5:
        return ("Corrective Adjustment Recommended", (0,80,180))
    return ("Increased Strain Risk", (0,0,180))

# -----------------------------
# Popup (Tkinter) — friendly feedback
# -----------------------------

def show_feedback_popup(checks, scores, mood_text):
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception:
        return False
    try:
        root = tk.Tk()
        root.title("Posture Feedback")
        # position popup to the right of the image window (best-effort)
        root.geometry("480x360+520+50")
        root.resizable(False, False)
        try:
            root.lift()
            root.attributes('-topmost', True)
            root.after(150, lambda: root.attributes('-topmost', False))
            root.deiconify()
            root.focus_force()
        except Exception:
            pass

        tk.Label(root, text=mood_text, font=("Segoe UI Emoji", 14)).pack(pady=(8, 4))

        meter_frame = tk.Frame(root)
        meter_frame.pack(fill='x', padx=12)
        for label, key in [("Neck", 'neck'), ("Back", 'back'), ("Head", 'head')]:
            row = tk.Frame(meter_frame)
            row.pack(fill='x', pady=6)
            tk.Label(row, text=label + ":", width=10, anchor='w').pack(side='left')
            pb = ttk.Progressbar(row, orient='horizontal', length=260, mode='determinate')
            pb.pack(side='left', padx=6)
            pb['value'] = scores.get(key, 0) * 10
            tk.Label(row, text=f"{scores.get(key,0)}/10", width=6).pack(side='left')

        ov_frame = tk.Frame(root)
        ov_frame.pack(fill='x', padx=12, pady=(4,0))
        ov = scores.get('overall10', 0)
        tk.Label(ov_frame, text=f"Overall: {ov}/10", font=("Segoe UI", 12, 'bold')).pack(anchor='w')

        # friendly header
        friendly = []
        if ov >= 8:
            friendly.append("Great job — your posture looks good! A couple of small tweaks will make it even better:")
        elif ov >= 5:
            friendly.append("You're on the right track — some adjustments will noticeably improve comfort and alignment:")
        else:
            friendly.append("Let's make a few focused changes to reduce strain. Start with the highest-impact items below:")

        # build suggestions and dedupe
        suggestions = []
        for key, (ok, msg) in checks.items():
            if key == 'overall' or ok is None:
                continue
            if not ok:
                if key == 'neck_flexion':
                    sc = None
                    if 'score=' in msg:
                        try:
                            sc = int(msg.split('score=')[1].split(')')[0])
                        except Exception:
                            sc = None
                    if sc is not None and sc <= 3:
                        suggestions.append('• Forward head — gently tuck your chin and raise the top of the screen so the center is near eye level.')
                    else:
                        suggestions.append('• Slight head forward — bring your head back a little and relax your shoulders.')
                elif key == 'trunk_sagittal':
                    suggestions.append('• Torso leans forward — sit fully back into your chair, support lower back, and keep feet grounded.')
                elif key == 'head_aligned':
                    suggestions.append('• Head offset — center your head over your shoulders; check that your monitor is directly in front.')
                elif key.endswith('elbow_90'):
                    suggestions.append('• Arm position — keep elbows close and bent near 90°, adjust chair height or keyboard.')
                elif key.endswith('knee_sitting'):
                    suggestions.append('• Leg position — adjust seat height so knees are near 110–135° or use a footrest.')

        seen = set()
        deduped = []
        for s in suggestions:
            if s not in seen:
                deduped.append(s)
                seen.add(s)

        # closing encouragement
        closing = []
        if ov >= 8:
            closing.append('Small, regular adjustments (microbreaks and a quick stretch every 30–45 minutes) will keep you comfortable.')
        elif ov >= 5:
            closing.append('Make these small changes and check if your comfort improves; small consistent adjustments often help.')
        else:
            closing.append('Start with the top 2 suggestions above and repeat checks after a few days; consistency helps a lot.')

        # display in text widget
        tips_frame = tk.Frame(root)
        tips_frame.pack(fill='both', expand=True, padx=12, pady=6)
        tk.Label(tips_frame, text='Feedback:', font=("Segoe UI", 10, 'bold')).pack(anchor='w')
        txt = tk.Text(tips_frame, wrap='word', height=8)
        txt.pack(fill='both', expand=True)

        for line in friendly:
            txt.insert('end', line + "\n")
        txt.insert('end', "\n")

        if deduped:
            for s in deduped:
                txt.insert('end', s + "\n")
        else:
            txt.insert('end', '• No specific corrections detected — keep moving and take breaks.\n')

        txt.insert('end', "\n")
        for line in closing:
            txt.insert('end', line + "\n")
        txt.config(state='disabled')

        tk.Button(root, text='Close', command=root.destroy).pack(pady=6)
        root.mainloop()
        return True
    except Exception:
        return False

# -----------------------------
# Static posture helper
# -----------------------------

def update_static_posture_counters(is_non_neutral, counters, frame_time_s, threshold_s=STATIC_POSTURE_THRESHOLD_S):
    if is_non_neutral:
        counters['non_neutral_accum'] = counters.get('non_neutral_accum', 0.0) + frame_time_s
    else:
        counters['non_neutral_accum'] = 0.0
    return counters['non_neutral_accum'] >= threshold_s

# -----------------------------
# Processing (image/video)
# -----------------------------

def process_image(image_path, save_annotated=False, show_popup=True, show_windows=True):
    if not os.path.exists(image_path):
        print('Image not found:', image_path)
        return
    frame = cv2.imread(image_path)
    if frame is None:
        print('Failed to read image:', image_path)
        return
    h, w = frame.shape[:2]

    features = None
    points = None
    if HAS_MEDIAPIPE:
        mp_pose = mp.solutions.pose
        with mp_pose.Pose(static_image_mode=True) as pose:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if getattr(res, 'pose_landmarks', None):
                out = features_mediapipe(res.pose_landmarks.landmark, w, h)
                if out is not None:
                    features, points = out
            else:
                out = features_face(frame)
                if out is not None:
                    features, points = out
    else:
        out = features_face(frame)
        if out is not None:
            features, points = out

    checks = evaluate_ergonomics(features, points, frame)
    annotated_clean = annotate_frame(frame, features, points, 'label')
    annotated = annotate_frame_with_meters(annotated_clean, features, points, 'label', checks)

    if show_windows:
        try:
            cv2.namedWindow('Annotated Image', cv2.WINDOW_NORMAL)
            cv2.moveWindow('Annotated Image', 10, 50)
            cv2.imshow('Annotated Image', annotated)
            cv2.waitKey(1)
        except Exception:
            pass

    scores = score_from_features(features)
    mood_text, _ = mood_from_scores(scores)
    popup_ok = False
    if show_popup:
        # Make sure image window is visible (brief topmost) and then show popup on the right
        try:
            try:
                cv2.setWindowProperty('Annotated Image', cv2.WND_PROP_TOPMOST, 1)
            except Exception:
                pass
        except Exception:
            pass
        popup_ok = show_feedback_popup(checks, scores, mood_text)
        try:
            cv2.setWindowProperty('Annotated Image', cv2.WND_PROP_TOPMOST, 0)
        except Exception:
            pass

    if not popup_ok:
        # fallback: draw tips on image and show until keypress
        annotated_fallback = annotated.copy()
        tips = []
        for key, (ok, msg) in checks.items():
            if key == 'overall':
                continue
            if ok is False and key in TIPS:
                tips.append(TIPS[key])
        if not tips:
            tips.append('Posture looks okay - keep changing positions frequently and take short breaks.')
        failed = sum(1 for v in checks.values() if v[0] is False)
        if failed >= 3:
            tips.append(EXTRA_TIPS[0])
        box_w = min(420, w - 40)
        box_h = min(160, h - 80)
        box_x = 20
        box_y = h - box_h - 20
        overlay = annotated_fallback.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (255,255,255), -1)
        alpha = 0.85
        cv2.addWeighted(overlay, alpha, annotated_fallback, 1 - alpha, 0, annotated_fallback)
        ty = box_y + 28
        cv2.putText(annotated_fallback, 'Posture Tips:', (box_x + 8, ty - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        for tip in tips[:6]:
            cv2.putText(annotated_fallback, u"- " + tip, (box_x + 10, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40,40,40), 1)
            ty += 20
        if show_windows:
            try:
                cv2.namedWindow('Annotated Image (with tips)', cv2.WINDOW_NORMAL)
                cv2.moveWindow('Annotated Image (with tips)', 10, 50)
                cv2.imshow('Annotated Image (with tips)', annotated_fallback)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except Exception:
                pass
    else:
        if show_windows:
            try:
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except Exception:
                pass


def process_video(video_path, save_annotated=True, show_popup=False, show_windows=False):
    if not os.path.exists(video_path):
        print('Video not found:', video_path)
        return
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('Failed to open video:', video_path)
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    base, _ = os.path.splitext(video_path)
    out_path = base + '_annotated.mp4'
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    neck_buf = deque(maxlen=SMOOTHING_WINDOW)
    back_buf = deque(maxlen=SMOOTHING_WINDOW)

    mp_pose = None
    if HAS_MEDIAPIPE:
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()
    else:
        pose = None

    frame_idx = 0
    aggregate = {}
    total_frames = 0
    last_features = None
    last_points = None
    static_counters = {'non_neutral_accum': 0.0}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        fh, fw = frame.shape[:2]
        features = None
        points = None

        if pose:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if getattr(res, 'pose_landmarks', None):
                out = features_mediapipe(res.pose_landmarks.landmark, fw, fh)
                if out is not None:
                    features, points = out
            else:
                out = features_face(frame)
                if out is not None:
                    features, points = out
        else:
            out = features_face(frame)
            if out is not None:
                features, points = out

        if features is not None:
            neck_buf.append(features.get('neck_flexion_deg', 0))
            back_buf.append(features.get('trunk_sagittal_deg', 0))
            smoothed = {
                'neck_flexion_deg': float(sum(neck_buf)) / len(neck_buf),
                'trunk_sagittal_deg': float(sum(back_buf)) / len(back_buf),
                'left_elbow_angle': features.get('left_elbow_angle'),
                'right_elbow_angle': features.get('right_elbow_angle'),
                'left_knee_angle': features.get('left_knee_angle'),
                'right_knee_angle': features.get('right_knee_angle'),
            }
        else:
            smoothed = None

        checks = evaluate_ergonomics(smoothed if smoothed is not None else features, points, frame)
        for k, (ok, msg) in checks.items():
            if k == 'overall':
                continue
            aggregate.setdefault(k, {'pass': 0, 'total': 0})
            if ok is True:
                aggregate[k]['pass'] += 1
            if ok is not None:
                aggregate[k]['total'] += 1
        total_frames += 1

        overall_ok = checks.get('overall', (False, ''))[0]
        frame_time_s = 1.0 / (fps or 25.0)
        is_non_neutral = not overall_ok
        static_exceeded = update_static_posture_counters(is_non_neutral, static_counters, frame_time_s)
        if static_exceeded and is_non_neutral and frame_idx % int(max(1, fps)) == 0:
            print(f"Static non-neutral posture detected for >= {STATIC_POSTURE_THRESHOLD_S} seconds (frame {frame_idx}). Consider taking a break.")

        label = 'good' if overall_ok else 'warning'
        annotated_clean = annotate_frame(frame, smoothed if smoothed is not None else features, points, label)
        annotated = annotate_frame_with_meters(annotated_clean, smoothed if smoothed is not None else features, points, label, checks)
        writer.write(annotated)

        if features is not None:
            last_features = features
            last_points = points

    cap.release()
    writer.release()
    print('Annotated video saved to:', out_path)

    report_path = base + '_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Ergonomics report summary for video: {video_path}\n")
        f.write(f"Frames processed: {total_frames}\n")
        for k, v in aggregate.items():
            if v['total'] == 0:
                pct = 'N/A'
            else:
                pct = f"{100.0 * v['pass'] / v['total']:.1f}%"
            f.write(f"{k}: {v['pass']} / {v['total']} frames passed ({pct})\n")
        f.write(f"Static non-neutral accum seconds (end): {static_counters.get('non_neutral_accum',0.0):.1f}s\n")
    print('Ergonomics report saved to:', report_path)

    if last_features is not None:
        cap2 = cv2.VideoCapture(out_path)
        ok, frame0 = cap2.read()
        cap2.release()
        if ok:
            annotated_clean = annotate_frame(frame0, last_features, last_points, 'label')
            annotated = annotate_frame_with_meters(annotated_clean, last_features, last_points, 'label', evaluate_ergonomics(last_features, last_points, frame0))
            if show_windows:
                try:
                    cv2.namedWindow('Annotated Video Snapshot', cv2.WINDOW_NORMAL)
                    cv2.moveWindow('Annotated Video Snapshot', 10, 50)
                    cv2.imshow('Annotated Video Snapshot', annotated)
                    cv2.waitKey(1)
                except Exception:
                    pass
            scores = score_from_features(last_features)
            mood_text, _ = mood_from_scores(scores)
            popup_ok = False
            if show_popup:
                popup_ok = show_feedback_popup(evaluate_ergonomics(last_features, last_points, frame0), scores, mood_text)
            if not popup_ok:
                if show_windows:
                    try:
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                    except Exception:
                        pass
            else:
                if show_windows:
                    try:
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                    except Exception:
                        pass

# -----------------------------
# Tests
# -----------------------------

def _test_angle_between_points():
    print('Running angle_between_points tests...')
    a = (0,0)
    b = (1,0)
    c = (1,1)
    ang = angle_between_points(a,b,c)
    assert abs(ang - 90.0) < 1e-6
    print('  Pass: right angle detected')


def _test_write_report():
    print('Running write_report test...')
    checks = {'neck': (True, 'ok'), 'overall': (True, 'ok')}
    p = 'test_report.txt'
    write_report(p, checks)
    assert os.path.exists(p)
    with open(p, 'r', encoding='utf-8') as f:
        data = f.read()
    os.remove(p)
    assert 'neck' in data and 'overall' in data
    print('  Pass: write_report created file and wrote keys')


def run_tests():
    _test_angle_between_points()
    _test_write_report()
    print('All quick tests done.')

# -----------------------------
# File picker
# -----------------------------

def prompt_and_pick_file():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        print('Opening file picker — choose an image.')
        filetypes = [('Image files', '.jpg *.jpeg *.png *.bmp'), ('All files', '*')]
        p = filedialog.askopenfilename(title='Select an image or video file', filetypes=filetypes)
        root.destroy()
        if not p:
            print('No file selected.')
            return None, None
        ext = os.path.splitext(p)[1].lower()
        if ext in ('.jpg', '.jpeg', '.png', '.bmp'):
            return 'image', p
        else:
            return 'video', p
    except Exception:
        if sys.stdin is None or not sys.stdin.isatty():
            print('No GUI available and stdin is not interactive. Please run the script with --image <path> or --video <path>.')
            return None, None
        print('No GUI available. Please type the full path to the file (or leave blank to cancel):')
        path = input().strip()
        if not path:
            return None, None
        ext = os.path.splitext(path)[1].lower()
        if ext in ('.jpg', '.jpeg', '.png', '.bmp'):
            return 'image', path
        else:
            return 'video', path

# -----------------------------
# CLI
# -----------------------------

def main():
    p = argparse.ArgumentParser(description='Posture detector for uploaded images & videos (popup mode)')
    p.add_argument('--image', type=str, help='Path to an image to process')
    p.add_argument('--video', type=str, help='Path to a video to process')
    p.add_argument('--test', action='store_true', help='Run internal self-tests and exit')
    p.add_argument('--show', action='store_true', help='Show annotated cv2 windows (explicit) ')
    p.add_argument('--popup', action='store_true', help='Show feedback popup (Tkinter) (explicit)')
    p.add_argument('--no-show', action='store_true', help='Do NOT show cv2 windows')
    p.add_argument('--no-popup', action='store_true', help='Do NOT show the Tkinter popup')
    args = p.parse_args()

    if args.test:
        run_tests()
        return

    provided_ui_flags = any([args.show, args.popup, args.no_show, args.no_popup])
    if provided_ui_flags:
        show_windows = bool(args.show) and not bool(args.no_show)
        show_popup = bool(args.popup) and not bool(args.no_popup)
    else:
        show_windows = True
        show_popup = True

    if args.image:
        process_image(args.image, save_annotated=False, show_popup=show_popup, show_windows=show_windows)
    elif args.video:
        process_video(args.video, save_annotated=False, show_popup=show_popup, show_windows=show_windows)
    else:
        ftype, path = prompt_and_pick_file()
        if ftype == 'image' and path:
            process_image(path, save_annotated=False, show_popup=show_popup, show_windows=show_windows)
        elif ftype == 'video' and path:
            process_video(path, save_annotated=False, show_popup=show_popup, show_windows=show_windows)
        else:
            print('No file processed. Goodbye.')


if __name__ == '__main__':
    main()
