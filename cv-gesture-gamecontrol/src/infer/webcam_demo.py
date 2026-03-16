import time
from collections import deque, Counter
from pathlib import Path

import cv2
import numpy as np
import joblib

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_TASK = Path("models/hand_landmarker.task")
CLF_PATH = Path("models/gesture_lr.joblib")

CONF_THRESH = 0.85          # ↑ plus strict pour éviter des faux gestes
MARGIN_THRESH = 0.25        # top1 - top2 doit être assez grand, sinon -> no_gesture
SMOOTH_N = 11                # vote sur 11 frames pour lisser


def normalize_landmarks_xy(xy42: np.ndarray) -> np.ndarray:
    xs = xy42[0::2].copy()
    ys = xy42[1::2].copy()
    x0, y0 = xs[0], ys[0]
    xs -= x0
    ys -= y0
    dmax = float(np.max(np.sqrt(xs * xs + ys * ys)))
    if dmax < 1e-6:
        dmax = 1.0
    xs /= dmax
    ys /= dmax
    out = np.empty_like(xy42)
    out[0::2] = xs
    out[1::2] = ys
    return out


def landmarks_to_features(hand_landmarks) -> np.ndarray | None:
    if hand_landmarks is None or len(hand_landmarks) != 21:
        return None
    feat = []
    for lm in hand_landmarks:
        feat.extend([lm.x, lm.y])
    feat = np.array(feat, dtype=np.float32)
    if feat.shape[0] != 42:
        return None
    return normalize_landmarks_xy(feat)


def majority_vote(labels: deque) -> str:
    if not labels:
        return "no_gesture"
    return Counter(labels).most_common(1)[0][0]


def draw_bbox_from_landmarks(frame, hand_landmarks):
    h, w = frame.shape[:2]
    xs = [lm.x for lm in hand_landmarks]
    ys = [lm.y for lm in hand_landmarks]
    x1 = max(0, int(min(xs) * w))
    y1 = max(0, int(min(ys) * h))
    x2 = min(w - 1, int(max(xs) * w))
    y2 = min(h - 1, int(max(ys) * h))

    pad = 12
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad); y2 = min(h - 1, y2 + pad)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


def main():
    assert MODEL_TASK.exists(), f"Introuvable: {MODEL_TASK}"
    assert CLF_PATH.exists(), f"Introuvable: {CLF_PATH}"

    clf = joblib.load(CLF_PATH)

    # Mode IMAGE : pas de timestamp -> beaucoup plus robuste sur Windows
    base_options = python.BaseOptions(model_asset_path=str(MODEL_TASK))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        running_mode=vision.RunningMode.IMAGE
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la webcam (index 0). Essaie 1 si besoin.")

    # Option : réduire la résolution pour gagner en stabilité/perf
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    smooth_buffer = deque(maxlen=SMOOTH_N)
    fps = 0.0
    prev = time.time()

    print("Appuie sur Q pour quitter.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = landmarker.detect(mp_image)

        raw_label = "no_gesture"
        conf = 0.0

        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            hand_lms = result.hand_landmarks[0]

            # bbox autour de la main
            draw_bbox_from_landmarks(frame, hand_lms)

            feats = landmarks_to_features(hand_lms)
            if feats is not None:
                proba = clf.predict_proba(feats.reshape(1, -1))[0]
                order = np.argsort(proba)[::-1]
                best_idx, second_idx = int(order[0]), int(order[1])
                best_conf = float(proba[best_idx])
                second_conf = float(proba[second_idx])
                margin = best_conf - second_conf

                best_label = clf.classes_[best_idx]

                # Gating + marge : si pas sûr -> no_gesture
                if (best_conf >= CONF_THRESH) and (margin >= MARGIN_THRESH):
                    raw_label = best_label
                    conf = best_conf
                else:
                    raw_label = "no_gesture"
                    conf = best_conf

        smooth_buffer.append(raw_label)
        label = majority_vote(smooth_buffer)

        now = time.time()
        dt = now - prev
        prev = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        text = f"{label} (raw={raw_label} conf={conf:.2f}) FPS={fps:.1f}"
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Gesture demo", frame)
        k = cv2.waitKey(1) & 0xFF
        if k in [ord("q"), ord("Q")]:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
