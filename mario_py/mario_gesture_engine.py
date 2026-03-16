# -*- coding: utf-8 -*-
import threading
import time
from collections import deque, Counter
from pathlib import Path

import cv2
import numpy as np
import joblib

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Fonctions pipeline (identiques à gesture_engine.py)

def _normalize_landmarks_xy(xy42: np.ndarray) -> np.ndarray:
    xs = xy42[0::2].copy()
    ys = xy42[1::2].copy()
    xs -= xs[0]; ys -= ys[0]
    dmax = float(np.max(np.sqrt(xs ** 2 + ys ** 2)))
    if dmax < 1e-6:
        dmax = 1.0
    xs /= dmax; ys /= dmax
    out = np.empty_like(xy42)
    out[0::2] = xs
    out[1::2] = ys
    return out


def _landmarks_to_features(hand_landmarks) -> np.ndarray | None:
    if hand_landmarks is None or len(hand_landmarks) != 21:
        return None
    feat = []
    for lm in hand_landmarks:
        feat.extend([lm.x, lm.y])
    feat = np.array(feat, dtype=np.float32)
    if feat.shape[0] != 42:
        return None
    return _normalize_landmarks_xy(feat)


def _majority_vote(buf: deque) -> str:
    if not buf:
        return "no_gesture"
    return Counter(buf).most_common(1)[0][0]


# GestureState

class GestureState:
    MARIO_GESTURES = {'palm', 'like', 'two_up', 'no_gesture'}

    def __init__(self):
        self._gesture = 'no_gesture'
        self._lock = threading.Lock()

    def set(self, gesture: str):
        with self._lock:
            self._gesture = gesture

    def get(self) -> str:
        with self._lock:
            return self._gesture


# ── GestureMarioEngine ────────────────────────────────────────────────────────

class GestureMarioEngine:
    def __init__(
        self,
        model_task: Path,
        clf_path: Path,
        conf_thresh: float = 0.85,
        margin_thresh: float = 0.25,
        smooth_n: int = 5,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        show_preview: bool = False,
    ):
        self.model_task    = Path(model_task)
        self.clf_path      = Path(clf_path)
        self.conf_thresh   = conf_thresh
        self.margin_thresh = margin_thresh
        self.smooth_n      = smooth_n
        self.camera_index  = camera_index
        self.width         = width
        self.height        = height
        self.show_preview  = show_preview
        self._stop         = False

    def stop(self):
        self._stop = True

    def run(self, state: GestureState):
        """
        Boucle principale du moteur CV.
        À appeler depuis un thread daemon — ne pas appeler depuis le thread Pygame.
        """
        assert self.model_task.exists(), f"Introuvable : {self.model_task}"
        assert self.clf_path.exists(),   f"Introuvable : {self.clf_path}"

        clf = joblib.load(self.clf_path)

        base_options = python.BaseOptions(model_asset_path=str(self.model_task))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            running_mode=vision.RunningMode.IMAGE,
        )
        landmarker = vision.HandLandmarker.create_from_options(options)

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError("Impossible d'ouvrir la webcam.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        smooth_buffer = deque(maxlen=self.smooth_n)

        while not self._stop:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_img)

            raw_label = "no_gesture"
            conf      = 0.0

            if result.hand_landmarks:
                feats = _landmarks_to_features(result.hand_landmarks[0])
                if feats is not None:
                    proba      = clf.predict_proba(feats.reshape(1, -1))[0]
                    order      = np.argsort(proba)[::-1]
                    best_idx   = int(order[0])
                    second_idx = int(order[1])
                    best_conf  = float(proba[best_idx])
                    margin     = best_conf - float(proba[second_idx])
                    best_label = clf.classes_[best_idx]

                    if best_conf >= self.conf_thresh and margin >= self.margin_thresh:
                        raw_label = best_label
                        conf      = best_conf

            smooth_buffer.append(raw_label)
            gesture = _majority_vote(smooth_buffer)

            if gesture not in GestureState.MARIO_GESTURES:
                gesture = "no_gesture"

            state.set(gesture)

            if self.show_preview:
                h, w = frame.shape[:2]

                if result.hand_landmarks:
                    _lms = result.hand_landmarks[0]
                    _xs  = [lm.x for lm in _lms]
                    _ys  = [lm.y for lm in _lms]
                    _pad = 0.04 
                    _x1  = max(0, int((min(_xs) - _pad) * w))
                    _y1  = max(0, int((min(_ys) - _pad) * h))
                    _x2  = min(w, int((max(_xs) + _pad) * w))
                    _y2  = min(h, int((max(_ys) + _pad) * h))
                    _bbox_color = (60, 220, 60) if gesture != 'no_gesture' else (60, 160, 220)
                    cv2.rectangle(frame, (_x1, _y1), (_x2, _y2), _bbox_color, 2)

                _ACTION = {'palm': 'DROITE', 'two_up': 'DROITE+SAUT',
                           'like': 'SAUT', 'no_gesture': 'stop'}
                _action  = _ACTION.get(gesture, '?')
                _color   = (80, 220, 80) if gesture != 'no_gesture' else (160, 160, 160)
                _overlay = frame.copy()
                cv2.rectangle(_overlay, (0, 0), (w, 60), (0, 0, 0), -1)
                cv2.addWeighted(_overlay, 0.45, frame, 0.55, 0, frame)
                cv2.putText(frame, f"{gesture}",
                            (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, _color, 2)
                cv2.putText(frame, f"-> {_action}   conf={conf:.2f}",
                            (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
                cv2.imshow("Mario gesture preview", frame)
                if cv2.waitKey(1) & 0xFF in [ord("q"), ord("Q")]:
                    break

        cap.release()
        if self.show_preview:
            cv2.destroyAllWindows()
