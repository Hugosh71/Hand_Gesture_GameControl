from pathlib import Path
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

GESTURES = ["two_up", "palm", "like", "ok", "point", "no_gesture"]

def load_csv(split: str):
    df = pd.read_csv(DATA_DIR / f"landmarks_{split}.csv")
    X = df.drop(columns=["label"]).values
    y = df["label"].values
    return X, y, df

def balance_with_oversample(df, target_per_class=6000, seed=42):
    parts = []
    for lab, d in df.groupby("label"):
        if len(d) >= target_per_class:
            parts.append(d.sample(n=target_per_class, random_state=seed))
        else:
            parts.append(d.sample(n=target_per_class, random_state=seed, replace=True))
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)

def downsample(df, max_per_class=5000, seed=42):
    # équilibrage simple : on prend au plus max_per_class exemples par classe
    parts = []
    for lab, d in df.groupby("label"):
        if len(d) > max_per_class:
            parts.append(d.sample(n=max_per_class, random_state=seed))
        else:
            parts.append(d)
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)

if __name__ == "__main__":
    # on équilibre surtout le train
    _, _, df_train = load_csv("train")
    df_train = df_train[df_train["label"].isin(GESTURES)]
    df_train_bal = balance_with_oversample(df_train, target_per_class=6000)


    X_train = df_train_bal.drop(columns=["label"]).values
    y_train = df_train_bal["label"].values

    X_val, y_val, _ = load_csv("val")
    X_test, y_test, _ = load_csv("test")

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
    max_iter=200,
    class_weight="balanced",
    solver="lbfgs"
))

    ])

    clf.fit(X_train, y_train)

    print("=== TRAIN (balanced) ===")
    pred = clf.predict(X_train)
    print(classification_report(y_train, pred))

    print("=== VAL ===")
    pred = clf.predict(X_val)
    print(classification_report(y_val, pred))

    print("=== TEST ===")
    pred = clf.predict(X_test)
    print(classification_report(y_test, pred))
    print("Confusion matrix (TEST):\n", confusion_matrix(y_test, pred, labels=clf.classes_))

    out_path = MODEL_DIR / "gesture_lr.joblib"
    joblib.dump(clf, out_path)
    print(f"[OK] Modèle sauvegardé -> {out_path}")
