# Reconnaissance gestuelle pour le contrôle de jeux vidéo

Projet final de Computer Vision @ Telecom Paris -  Reconnaissance de gestes de la main en temps réel à partir d'une webcam standard, appliquée au contrôle de deux jeux vidéo : **Flappy Bird** et **Super Mario Bros (Python)**.

Pipeline : MediaPipe HandLandmarker → features landmarks 42D normalisées → classifieur de régression logistique → dispatcher temps réel.

---

## Structure du dépôt

```
Hand_gesture_GameControl/
│
├── cv-gesture-gamecontrol/               # Dossier principal du projet
│   ├── HandGesture_FANCHINI_CINTRA_ZHANG.ipynb   # Notebook principal (rendu)
│   ├── gesture_creation.py               # Capture des données propres (webcam)
│   ├── extract_own_landmarks.py          # Extraction landmarks depuis images propres
│   ├── scripts/
│   │   └── build_landmarks_csv.py        # Extraction landmarks sur HaGRID → CSV
│   ├── models/
│   │   ├── gesture_lr_baseline.joblib    # Classifieur baseline (entraîné sur HaGRID)
│   │   ├── gesture_lr_adapted.joblib     # Classifieur adapté (HaGRID + données propres)
│   │   └── gesture_lr.joblib             # Alias du modèle courant
│   │   # hand_landmarker.task            # ← non inclus, téléchargé automatiquement
│   └── data/
│       ├── own/
│       │   └── images/                   # Images webcam collectées (incluses)
│       └── processed/
│           └── landmarks_own.csv         # Landmarks des données propres collectées
│           # landmarks_train/val/test/combined.csv  ← générés automatiquement par le notebook
│
├── flappy_py/
│   ├── flappy.py                         # Jeu Flappy Bird avec contrôle gestuel
│   └── gesture_engine.py                 # Moteur d'inférence temps réel (Flappy)
│
├── mario_py/
│   ├── mario_gesture.py                  # Intégration Mario (GestureInput, run_mario_game)
│   └── mario_gesture_engine.py           # Moteur CV Mario (état continu, thread-safe)
│
├── super-mario-python/                   # Code source du jeu Mario (Python/Pygame)
│
├── requirements.txt
└── README.md
```

---

## Prérequis

- Python **3.10** ou **3.11** — MediaPipe n'est pas encore compatible avec Python 3.12+
- Une webcam fonctionnelle (pour les démos locales)
- Environ **3 GB** d'espace disque libre (annotations HaGRID ~680 MB + CSVs générés ~220 MB + marge)

> **Utilisateurs Mac** : voir la section [Problèmes connus sur Mac](#problèmes-connus-sur-mac) avant d'installer.

---

## Installation

```bash
# Cloner le dépôt
git clone <url-du-repo>
cd Hand_gesture_VG

# Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

# Installer les dépendances
pip install -r requirements.txt
```

---

## Lancer le notebook

```bash
cd cv-gesture-gamecontrol
# jupyter notebook HandGesture_FANCHINI_CINTRA_ZHANG.ipynb si usage hors vscode
```

En exécutant les cellules dans l'ordre, le notebook gère automatiquement toute la chaîne de préparation :

1. **Section 0** — télécharge `hand_landmarker.task` (~7.5 MB) depuis les serveurs MediaPipe si absent.
2. **Section 2** — télécharge et extrait les annotations HaGRID (~680 MB), puis génère les CSVs de landmarks via `build_landmarks_csv.py` (~5–15 min).
3. **Sections 3–10** — entraînement, évaluation, adaptation, démos — s'exécutent normalement.

La première exécution complète prend **30–40 minutes** (téléchargements + génération des CSVs). Les exécutions suivantes sautent les étapes déjà effectuées.

---

## Fichiers non inclus dans le dépôt (gérés automatiquement)

| Fichier / Dossier | Taille | Obtenu automatiquement par |
|---|---|---|
| `models/hand_landmarker.task` | ~7.5 MB | Cellule Section 0 — téléchargement MediaPipe |
| `data/raw/hagrid/annotations.zip` | ~680 MB | Cellule Section 2 — téléchargement HaGRID |
| `data/processed/landmarks_train.csv` | ~95 MB | Cellule Section 2 — `build_landmarks_csv.py` |
| `data/processed/landmarks_val.csv` | ~12 MB | Cellule Section 2 — `build_landmarks_csv.py` |
| `data/processed/landmarks_test.csv` | ~20 MB | Cellule Section 2 — `build_landmarks_csv.py` |

Les fichiers suivants **sont inclus** dans le dépôt :
- `models/gesture_lr_baseline.joblib`, `gesture_lr_adapted.joblib`, `gesture_lr.joblib` (~4.5 KB chacun) — classifieurs entraînés, nécessaires pour les démos
- `data/processed/landmarks_own.csv` — landmarks des données propres collectées
- `data/own/images/` — images webcam utilisées pour l'adaptation (Section 7)

---

## Reproductibilité

Le notebook est entièrement reproductible à partir du dépôt seul, sans configuration manuelle. Chaque cellule de téléchargement vérifie si les fichiers sont déjà présents avant d'agir — relancer le notebook après une première exécution complète est sans effet.

---

## Démos locales (optionnel)

Les démos nécessitent une webcam et les classifieurs `.joblib` (inclus).

### Flappy Bird gestuel

```python
# Depuis le notebook, cellule optionnelle de la Section 5
# Ou directement :
from pathlib import Path
import sys
sys.path.insert(0, 'flappy_py')
from flappy import run_flappy_game
run_flappy_game(
    model_task=Path('cv-gesture-gamecontrol/models/hand_landmarker.task'),
    clf_path=Path('cv-gesture-gamecontrol/models/gesture_lr_adapted.joblib'),
    show_preview=True,   # mettre False sur Mac (cv2.imshow interdit depuis un thread secondaire)
)
```

Geste : `like` (👍) → battement d'ailes.

### Mario gestuel

```python
# Depuis le notebook, cellule optionnelle de la Section 9 (RUN_MARIO_DEMO = True)
# Ou directement :
from pathlib import Path
import sys
sys.path.insert(0, 'mario_py')
from mario_gesture import run_mario_game
run_mario_game(
    model_task=Path('cv-gesture-gamecontrol/models/hand_landmarker.task'),
    clf_path=Path('cv-gesture-gamecontrol/models/gesture_lr_adapted.joblib'),
    show_preview=True,   # mettre False sur Mac (cv2.imshow interdit depuis un thread secondaire)
)
```

Gestes : `palm` (👋) → courir à droite · `like` (👍) → sauter · `two_up` (✌) → sauter en courant.

---

## Scripts externes

Le notebook constitue le support principal du rendu. Il s'appuie sur plusieurs scripts Python qui ont été utilisés lors du développement réel du projet mais ne sont pas reproduits intégralement dans le notebook :

| Script | Rôle |
|---|---|
| `scripts/build_landmarks_csv.py` | Extraction des landmarks MediaPipe sur HaGRID → CSV |
| `gesture_creation.py` | Capture interactive des données propres par webcam |
| `extract_own_landmarks.py` | Extraction des landmarks depuis les images propres collectées |
| `gesture_engine.py` | Moteur d'inférence temps réel pour Flappy Bird |
| `flappy.py` | Intégration Flappy Bird avec contrôle gestuel |
| `mario_gesture_engine.py` | Moteur CV pour Mario (état gestuel continu, thread-safe) |
| `mario_gesture.py` | Intégration Mario : `GestureInput`, `run_mario_game()` |

Les choix de conception de ces scripts sont documentés et analysés dans les sections correspondantes du notebook.

---

## Problèmes connus sur Mac

### pygame — `SDL.h file not found`

Sur macOS, pygame nécessite SDL installé via Homebrew. Sans ça, la compilation échoue avec :
```
fatal error: 'SDL.h' file not found
```

**Fix :**
```bash
brew install sdl2 sdl2_image sdl2_mixer sdl2_ttf
pip install pygame
```
Si Homebrew n'est pas installé : [https://brew.sh](https://brew.sh)

### Python 3.12+ incompatible avec MediaPipe

MediaPipe ne supporte pas encore Python 3.12 ou supérieur. Si votre Mac tourne sur Python 3.14 (ou 3.12/3.13), l'installation de `mediapipe` échouera silencieusement ou produira des erreurs à l'exécution.

**Fix — installer Python 3.11 et créer un environnement dédié :**
```bash
brew install python@3.11
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Webcam inaccessible depuis le notebook (`not authorized to capture video`)

Sur macOS, OpenCV ne peut pas accéder à la caméra depuis un thread secondaire (ce que fait notre architecture à deux threads). L'erreur typique :
```
OpenCV: not authorized to capture video (status 0)
OpenCV: can not spin main run loop from other thread
RuntimeError: Impossible d'ouvrir la webcam.
```

**Fix — lancer Jupyter avec la variable d'environnement suivante :**
```bash
OPENCV_AVFOUNDATION_SKIP_AUTH=1 
```
macOS affichera une popup de permission caméra à la première exécution — accepter.

### Preview webcam (`cv2.error: Unknown C++ exception`) — Flappy Bird / Mario

Sur macOS, `cv2.imshow()` ne peut pas être appelé depuis un thread secondaire (contrainte Cocoa). Notre moteur gestuel tourne dans un thread daemon — la preview caméra plante donc systématiquement sur Mac.

**Fix — désactiver la preview dans les cellules de démo :**
```python
# Flappy Bird
run_flappy_game(model_task=_MODEL_TASK, clf_path=_CLF_PATH, show_preview=False)

# Mario
run_mario_game(model_task=_MODEL_TASK, clf_path=_CLF_PATH, show_preview=False)
```
Le jeu fonctionne normalement, seule la fenêtre de visualisation de la webcam est désactivée.

### Conflits SDL (`Class SDLApplication is implemented in both cv2 and pygame`)

`opencv-python` et `pygame` embarquent chacun leur propre copie de SDL, ce qui génère des warnings au démarrage. Ces warnings sont inoffensifs mais peuvent provoquer des crashs aléatoires.

**Fix 1 — remplacer `opencv-python` par la version sans interface graphique (recommandé) :**
```bash
pip uninstall opencv-python
pip install opencv-python-headless
```

**Fix 2 — si `pip install opencv-python-headless` échoue (réseau ou autre) :**
Supprimer manuellement la copie de SDL embarquée par cv2 :
```bash
rm .venv/lib/python3.11/site-packages/cv2/.dylibs/libSDL2-2.0.0.dylib
```
(Remplacer `python3.11` par votre version si nécessaire.)

### Impossible d'interrompre une cellule pygame depuis Jupyter

Quand une démo Flappy Bird ou Mario est lancée depuis le notebook, le bouton "Interrupt Kernel" de Jupyter peut ne pas arrêter la cellule (la fenêtre pygame reste ouverte).

**Fix — fermer la fenêtre pygame en cliquant le bouton X.** Cela déclenche l'événement `pygame.QUIT` et la fonction retourne proprement. En dernier recours, Ctrl+C dans le terminal qui a lancé Jupyter fonctionne aussi.

**En résumé pour Mac, après l'installation initiale :**
```bash
pip uninstall opencv-python
pip install opencv-python-headless
OPENCV_AVFOUNDATION_SKIP_AUTH=1 jupyter notebook
```

---

## Auteurs

HUGO FANCHINI · PAUL CINTRA · YIMOU ZHANG
