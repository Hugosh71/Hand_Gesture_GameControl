import json
import csv
from pathlib import Path
from tqdm import tqdm

GESTURES = ["two_up", "palm", "like", "ok", "point", "no_gesture"]

RAW_ANN_DIR = Path("data/raw/hagrid/annotations")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def flatten_landmarks(lms):
    """
    lms attendu: liste de 21 points [[x,y], ...]
    retourne: liste [x0,y0,x1,y1,...] longueur 42
    """
    if not isinstance(lms, list) or len(lms) != 21:
        return None

    feat = []
    for pt in lms:
        if (not isinstance(pt, list)) or len(pt) < 2:
            return None
        x, y = pt[0], pt[1]
        try:
            feat.extend([float(x), float(y)])
        except Exception:
            return None

    if len(feat) != 42:
        return None
    return feat

def normalize_landmarks(feat):
    """
    Normalisation robuste :
    - recentrage poignet (point 0)
    - normalisation par la distance max
    """
    if feat is None or len(feat) != 42:
        return None

    xs = feat[0::2]
    ys = feat[1::2]
    if len(xs) == 0 or len(ys) == 0:
        return None

    x0, y0 = xs[0], ys[0]
    xs = [x - x0 for x in xs]
    ys = [y - y0 for y in ys]

    dmax = 0.0
    for x, y in zip(xs, ys):
        d = (x * x + y * y) ** 0.5
        if d > dmax:
            dmax = d
    if dmax < 1e-6:
        dmax = 1.0

    out = []
    for x, y in zip(xs, ys):
        out.extend([x / dmax, y / dmax])
    return out  # longueur 42

def build_split(split: str):
    rows = []
    skipped = 0

    for g in GESTURES:
        p = RAW_ANN_DIR / split / f"{g}.json"
        if not p.exists():
            print(f"[WARN] {p} introuvable -> geste ignoré")
            continue

        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)

        for _, item in tqdm(data.items(), desc=f"{split}:{g}"):
            lms_list = item.get("hand_landmarks")
            labels = item.get("labels")

            if not isinstance(lms_list, list) or not isinstance(labels, list):
                skipped += 1
                continue

            n = min(len(lms_list), len(labels))
            if n == 0:
                skipped += 1
                continue

            for i in range(n):
                lms = lms_list[i]
                lab = labels[i]
                if lab != g:
                    continue

                feat = flatten_landmarks(lms)
                feat = normalize_landmarks(feat)
                if feat is None:
                    skipped += 1
                    continue

                rows.append((feat, lab))

    out_path = OUT_DIR / f"landmarks_{split}.csv"
    header = [f"f{i}" for i in range(42)] + ["label"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for feat, lab in rows:
            w.writerow(feat + [lab])

    print(f"[OK] {split}: {len(rows)} samples -> {out_path} | skipped={skipped}")

if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        build_split(split)
