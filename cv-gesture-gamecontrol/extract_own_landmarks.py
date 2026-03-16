import csv
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from tqdm import tqdm


IMAGE_ROOT = Path("data/own/images")
OUTPUT_CSV = Path("data/processed/landmarks_own.csv")
MODEL_PATH = Path("models/hand_landmarker.task")

GESTURE_CLASSES = [
    "like",
    "no_gesture",
    "ok",
    "palm",
    "point",
    "two_up",
]

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}

def normalize_landmarks_xy(xy42: np.ndarray) -> np.ndarray | None:
    """
    xy42 : [x0,y0,x1,y1,...,x20,y20]
    Normalisation :
    - centrage sur le poignet
    - division par la distance max au poignet
    """
    if xy42.shape[0] != 42:
        return None

    xs = xy42[0::2].copy()
    ys = xy42[1::2].copy()

    x0, y0 = xs[0], ys[0]
    xs -= x0
    ys -= y0

    dmax = float(np.max(np.sqrt(xs**2 + ys**2)))
    if dmax < 1e-6:
        return None

    xs /= dmax
    ys /= dmax

    out = np.empty_like(xy42)
    out[0::2] = xs
    out[1::2] = ys
    return out


def landmarks_to_feature_vector(hand_landmarks) -> np.ndarray | None:
    """
    Convertit 21 landmarks MediaPipe -> vecteur 42D normalisé
    """
    if hand_landmarks is None or len(hand_landmarks) != 21:
        return None

    feat = []
    for lm in hand_landmarks:
        feat.extend([lm.x, lm.y])

    feat = np.array(feat, dtype=np.float32)
    return normalize_landmarks_xy(feat)


def build_landmarker(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Fichier modèle introuvable : {model_path}\n"
            "Place hand_landmarker.task dans le dossier models/."
        )

    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        running_mode=mp_vision.RunningMode.IMAGE,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def process_image(image_path: Path, landmarker) -> np.ndarray | None:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        return None

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    result = landmarker.detect(mp_image)

    if not result.hand_landmarks or len(result.hand_landmarks) == 0:
        return None

    return landmarks_to_feature_vector(result.hand_landmarks[0])


def main():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    landmarker = build_landmarker(MODEL_PATH)

    rows = []
    stats = {cls: {"total": 0, "kept": 0, "failed": 0} for cls in GESTURE_CLASSES}

    for gesture in GESTURE_CLASSES:
        class_dir = IMAGE_ROOT / gesture
        if not class_dir.exists():
            print(f"[WARN] Dossier absent : {class_dir}")
            continue

        image_paths = sorted(
            [p for p in class_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS]
        )

        for img_path in tqdm(image_paths, desc=f"Classe {gesture}"):
            stats[gesture]["total"] += 1

            feat = process_image(img_path, landmarker)
            if feat is None:
                stats[gesture]["failed"] += 1
                continue

            rows.append(list(feat) + [gesture, str(img_path)])
            stats[gesture]["kept"] += 1

    header = [f"f{i}" for i in range(42)] + ["label", "source_path"]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print("\nExtraction terminée.")
    print(f"CSV sauvegardé : {OUTPUT_CSV}\n")

    print("Résumé par classe :")
    total_total = total_kept = total_failed = 0
    for gesture in GESTURE_CLASSES:
        s = stats[gesture]
        total_total += s["total"]
        total_kept += s["kept"]
        total_failed += s["failed"]
        print(
            f"  {gesture:<12} total={s['total']:<4}  "
            f"kept={s['kept']:<4}  failed={s['failed']:<4}"
        )

    print(
        f"\nTOTAL  total={total_total}  kept={total_kept}  failed={total_failed}"
    )


if __name__ == "__main__":
    main()