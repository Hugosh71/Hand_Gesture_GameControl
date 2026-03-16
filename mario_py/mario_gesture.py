# -*- coding: utf-8 -*-
import sys
import os
import time
import threading
from pathlib import Path

import pygame
from pygame.locals import K_ESCAPE, K_F5

# Chemins par défaut

_HERE      = Path(__file__).parent.resolve()
_MARIO_DIR = (_HERE / '../super-mario-python/super-mario-python-master').resolve()

_DEFAULT_MODEL_TASK = (_HERE / '../cv-gesture-gamecontrol/models/hand_landmarker.task').resolve()
_DEFAULT_CLF_PATH   = (_HERE / '../cv-gesture-gamecontrol/models/gesture_lr_adapted.joblib').resolve()


#GestureInput : remplace classes/Input.py 

class GestureInput:

    JUMP_COOLDOWN = 0.4  # secondes minimum entre deux sauts gestuels

    def __init__(self, entity, gesture_state):
        self.entity = entity
        self._gs    = gesture_state
        self._last_jump = 0.0

    # Interface publique (identique à classes/Input.py)
    def checkForInput(self):
        events = pygame.event.get()
        self.checkForKeyboardInput()
        self.checkForQuitAndRestartInputEvents(events)

    def checkForKeyboardInput(self):
        gesture = self._gs.get()

        # Direction continue (hold) + combo droite+saut
        if gesture in ('palm', 'two_up'):
            self.entity.traits['goTrait'].direction = 1
        else:
            self.entity.traits['goTrait'].direction = 0

        # Saut impulsif avec cooldown
        # like    = saut seul
        # two_up  = saut + droite (combo)
        now = time.time()
        if gesture in ('like', 'two_up') and (now - self._last_jump) >= self.JUMP_COOLDOWN:
            self.entity.traits['jumpTrait'].jump(True)
            self._last_jump = now
        else:
            self.entity.traits['jumpTrait'].jump(False)

        # Pas d'accélération automatique
        self.entity.traits['goTrait'].boost = False

    def checkForQuitAndRestartInputEvents(self, events):
        for event in events:
            if event.type == pygame.QUIT:
                # Signale l'arrêt propre sans sys.exit()
                self.entity.restart = True
                return
            if event.type == pygame.KEYDOWN and (
                event.key == K_ESCAPE or event.key == K_F5
            ):
                self.entity.pause = True
                self.entity.pauseObj.createBackgroundBlur()

    # Conservée pour compatibilité avec la base Input
    def checkForMouseInput(self, events):
        pass


# run_mario_game

def run_mario_game(
    model_task=None,
    clf_path=None,
    show_preview: bool = False,
):
    model_task = Path(model_task or _DEFAULT_MODEL_TASK)
    clf_path   = Path(clf_path   or _DEFAULT_CLF_PATH)

    assert model_task.exists(), f"Modèle MediaPipe introuvable : {model_task}"
    assert clf_path.exists(),   f"Classifieur introuvable : {clf_path}"

    # Ajouter mario_py/ (ce dossier) au path pour mario_gesture_engine
    _mario_py = str(_HERE)
    if _mario_py not in sys.path:
        sys.path.insert(0, _mario_py)

    # Ajouter le dossier Mario au path pour ses imports internes
    _mario_src = str(_MARIO_DIR)
    if _mario_src not in sys.path:
        sys.path.insert(0, _mario_src)

    from mario_gesture_engine import GestureState, GestureMarioEngine

    # chdir requis : Mario utilise des chemins relatifs pour ses assets
    _prev_dir = os.getcwd()
    os.chdir(_MARIO_DIR)

    try:
        from classes.Dashboard import Dashboard
        from classes.Level import Level
        from classes.Menu import Menu
        from classes.Sound import Sound
        from entities.Mario import Mario as MarioEntity

        # Démarrage du moteur gestuel (thread daemon) 
        state  = GestureState()
        engine = GestureMarioEngine(
            model_task=model_task,
            clf_path=clf_path,
            show_preview=show_preview,
        )
        cv_thread = threading.Thread(target=engine.run, args=(state,), daemon=True)
        cv_thread.start()
        print("Moteur gestuel démarré (thread CV).")
        print("Gestes : palm=droite | two_up=gauche | like=saut | (rien)=stop")
        print("Démarrez le jeu avec le clavier (Entrée / Espace dans le menu).")

        # Initialisation Pygame 
        pygame.mixer.pre_init(44100, -16, 2, 4096)
        pygame.init()
        screen    = pygame.display.set_mode((640, 480))
        dashboard = Dashboard("./img/font.png", 8, screen)
        sound     = Sound()
        level     = Level(screen, sound, dashboard)
        menu      = Menu(screen, dashboard, level, sound)

        # Menu (clavier : Entrée ou Espace pour démarrer)
        while not menu.start:
            menu.update()

        mario = MarioEntity(0, 0, level, screen, dashboard, sound)
        clock = pygame.time.Clock()

        # Injection du contrôle gestuel 
        mario.input = GestureInput(mario, state)

        # Boucle de jeu 
        while not mario.restart:
            pygame.display.set_caption(
                f"Mario gestuel  |  {state.get():<12}  |  {int(clock.get_fps())} FPS"
            )
            if mario.pause:
                mario.pauseObj.update()
            else:
                level.drawLevel(mario.camera)
                dashboard.update()
                mario.update()
            pygame.display.update()
            clock.tick(60)

    except KeyboardInterrupt:
        pass

    finally:
        engine.stop()
        pygame.quit()
        os.chdir(_prev_dir)
        print("Mario gestuel arrêté.")


if __name__ == "__main__":
    run_mario_game()
