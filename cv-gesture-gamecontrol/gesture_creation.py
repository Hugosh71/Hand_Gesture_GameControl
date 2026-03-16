import cv2
import time
from pathlib import Path

BASE_DIR = Path("data/own/images")

CLASSES = {
    "1": "like",
    "2": "no_gesture",
    "3": "ok",
    "4": "palm",
    "5": "point",
    "6": "two_up",
}

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

SAVE_INTERVAL_SEC = 0.18

MIRROR_VIEW = True

for class_name in CLASSES.values():
    (BASE_DIR / class_name).mkdir(parents=True, exist_ok=True)

def next_filename(class_dir: Path, prefix: str) -> Path:
    existing = sorted(class_dir.glob(f"{prefix}_*.jpg"))
    if not existing:
        idx = 1
    else:
        last = existing[-1].stem.split("_")[-1]
        idx = int(last) + 1 if last.isdigit() else len(existing) + 1
    return class_dir / f"{prefix}_{idx:04d}.jpg"

def draw_help(frame, current_class, recording, counts):
    lines = [
        f"Classe courante : {current_class}",
        f"Enregistrement : {'ON' if recording else 'OFF'}",
        "Touches :",
        "  1=like  2=no_gesture  3=ok  4=palm  5=point  6=two_up",
        "  r = start/stop enregistrement",
        "  s = snapshot unique",
        "  q = quitter",
        "",
        f"Compteurs : " + " | ".join([f"{k}:{counts[v]}" for k, v in CLASSES.items()]),
    ]

    y = 25
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la webcam.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    current_class = "like"
    recording = False
    last_save_time = 0.0

    counts = {}
    for class_name in CLASSES.values():
        counts[class_name] = len(list((BASE_DIR / class_name).glob("*.jpg")))

    print("Capture dataset gestes")
    print("1=like  2=no_gesture  3=ok  4=palm  5=point  6=two_up")
    print("r = start/stop enregistrement | s = snapshot | q = quitter")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        if MIRROR_VIEW:
            frame = cv2.flip(frame, 1)

        display = frame.copy()
        draw_help(display, current_class, recording, counts)

        now = time.time()

        # Enregistrement automatique en rafale
        if recording and (now - last_save_time >= SAVE_INTERVAL_SEC):
            class_dir = BASE_DIR / current_class
            out_path = next_filename(class_dir, current_class)
            cv2.imwrite(str(out_path), frame)
            counts[current_class] += 1
            last_save_time = now

            cv2.putText(display, f"Saved: {out_path.name}", (10, FRAME_HEIGHT - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Dataset capture", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("r"):
            recording = not recording
            if recording:
                last_save_time = 0.0
                print(f"[ON] Enregistrement pour la classe : {current_class}")
            else:
                print(f"[OFF] Enregistrement stoppe pour : {current_class}")
        elif key == ord("s"):
            class_dir = BASE_DIR / current_class
            out_path = next_filename(class_dir, current_class)
            cv2.imwrite(str(out_path), frame)
            counts[current_class] += 1
            print(f"[SNAPSHOT] {out_path}")
        elif chr(key) in CLASSES:
            current_class = CLASSES[chr(key)]
            print(f"[CLASS] Classe courante -> {current_class}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()