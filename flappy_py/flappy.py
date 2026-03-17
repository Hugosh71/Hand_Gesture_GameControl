import pygame, sys, random
import threading
from gesture_engine import GestureFlapEngine
from pathlib import Path


# Tuning
PIPE_SPEED = 4
FLOOR_SPEED = 3
GRAVITY = 0.21
FLAP_STRENGTH = 7
GAME_FPS = 60

SHOW_CAMERA_PREVIEW = True

_HERE = Path(__file__).parent.resolve()
_DEFAULT_MODEL_TASK = (_HERE / "../cv-gesture-gamecontrol/models/hand_landmarker.task").resolve()
_DEFAULT_CLF_PATH   = (_HERE / "../cv-gesture-gamecontrol/models/gesture_lr.joblib").resolve()



def draw_floor():
	screen.blit(floor_surface,(floor_x_pos,900))
	screen.blit(floor_surface,(floor_x_pos + 576,900))

def create_pipe():
	random_pipe_pos = random.choice(pipe_height)
	bottom_pipe = pipe_surface.get_rect(midtop = (700,random_pipe_pos))
	top_pipe = pipe_surface.get_rect(midbottom = (700,random_pipe_pos - 300))
	return bottom_pipe,top_pipe

def move_pipes(pipes):
	for pipe in pipes:
		pipe.centerx -= PIPE_SPEED
	return pipes

def draw_pipes(pipes):
	for pipe in pipes:
		if pipe.bottom >= 1024:
			screen.blit(pipe_surface,pipe)
		else:
			flip_pipe = pygame.transform.flip(pipe_surface,False,True)
			screen.blit(flip_pipe,pipe)

def remove_pipes(pipes):
	return [pipe for pipe in pipes if pipe.right > -50]

def check_collision(pipes):
	for pipe in pipes:
		if bird_rect.colliderect(pipe):
			death_sound.play()
			return False
	if bird_rect.top <= -100 or bird_rect.bottom >= 900:
		return False
	return True

def rotate_bird(bird):
	new_bird = pygame.transform.rotozoom(bird,-bird_movement * 3,1)
	return new_bird

def bird_animation():
	new_bird = bird_frames[bird_index]
	new_bird_rect = new_bird.get_rect(center = (100,bird_rect.centery))
	return new_bird,new_bird_rect

def score_display(game_state):
	if game_state == 'main_game':
		score_surface = game_font.render(str(int(score)),True,(255,255,255))
		score_rect = score_surface.get_rect(center = (288,100))
		screen.blit(score_surface,score_rect)
	if game_state == 'game_over':
		score_surface = game_font.render(f'Score: {int(score)}' ,True,(255,255,255))
		score_rect = score_surface.get_rect(center = (288,100))
		screen.blit(score_surface,score_rect)
		high_score_surface = game_font.render(f'High score: {int(high_score)}',True,(255,255,255))
		high_score_rect = high_score_surface.get_rect(center = (288,850))
		screen.blit(high_score_surface,high_score_rect)

def update_score(score, high_score):
	if score > high_score:
		high_score = score
	return high_score


def run_flappy_game(model_task=None, clf_path=None):
	global screen, floor_surface, floor_x_pos, pipe_surface, pipe_height
	global bird_rect, bird_movement, bird_frames, bird_index, bird_surface
	global game_font, score, high_score, death_sound

	model_task = Path(model_task) if model_task is not None else _DEFAULT_MODEL_TASK
	clf_path   = Path(clf_path)   if clf_path   is not None else _DEFAULT_CLF_PATH

	pygame.mixer.pre_init(frequency=44100, size=16, channels=1, buffer=512)
	pygame.init()
	screen = pygame.display.set_mode((576,1024))
	clock = pygame.time.Clock()
	game_font = pygame.font.Font('04B_19.ttf',40)

	gravity = GRAVITY
	bird_movement = 0
	game_active = True
	score = 0
	high_score = 0

	bg_surface = pygame.image.load('assets/background-day.png').convert()
	bg_surface = pygame.transform.scale2x(bg_surface)

	floor_surface = pygame.image.load('assets/base.png').convert()
	floor_surface = pygame.transform.scale2x(floor_surface)
	floor_x_pos = 0

	bird_downflap = pygame.transform.scale2x(pygame.image.load('assets/bluebird-downflap.png').convert_alpha())
	bird_midflap  = pygame.transform.scale2x(pygame.image.load('assets/bluebird-midflap.png').convert_alpha())
	bird_upflap   = pygame.transform.scale2x(pygame.image.load('assets/bluebird-upflap.png').convert_alpha())
	bird_frames = [bird_downflap, bird_midflap, bird_upflap]
	bird_index = 0
	bird_surface = bird_frames[bird_index]
	bird_rect = bird_surface.get_rect(center=(100,512))

	BIRDFLAP = pygame.USEREVENT + 1
	pygame.time.set_timer(BIRDFLAP,200)

	pipe_surface = pygame.image.load('assets/pipe-green.png')
	pipe_surface = pygame.transform.scale2x(pipe_surface)
	pipe_list = []
	SPAWNPIPE = pygame.USEREVENT
	pygame.time.set_timer(SPAWNPIPE,1800)
	pipe_height = [400,600,800]

	game_over_surface = pygame.transform.scale2x(pygame.image.load('assets/message.png').convert_alpha())
	game_over_rect = game_over_surface.get_rect(center=(288,512))

	flap_sound  = pygame.mixer.Sound('sound/sfx_wing.wav')
	death_sound = pygame.mixer.Sound('sound/sfx_hit.wav')
	score_sound = pygame.mixer.Sound('sound/sfx_point.wav')
	score_sound_countdown = 100

	GESTURE_FLAP = pygame.USEREVENT + 50
	flap_flag = threading.Event()

	def set_flap_flag():
		flap_flag.set()

	gesture_engine = GestureFlapEngine(
		model_task=model_task,
		clf_path=clf_path,
		trigger_gesture="like",
		conf_thresh=0.90,
		margin_thresh=0.30,
		smooth_n=9,
		cooldown_sec=0.20,
		show_preview=SHOW_CAMERA_PREVIEW
	)

	gesture_thread = threading.Thread(
		target=gesture_engine.run,
		args=(set_flap_flag,),
		daemon=True
	)
	gesture_thread.start()

	# Boucle principale
	try:
		while True:
			if flap_flag.is_set():
				flap_flag.clear()
				pygame.event.post(pygame.event.Event(GESTURE_FLAP))

			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					gesture_engine.stop()
					pygame.quit()
					return

				if event.type == pygame.KEYDOWN:
					if event.key == pygame.K_SPACE and game_active:
						bird_movement = 0
						bird_movement -= FLAP_STRENGTH
						flap_sound.play()
					if event.key == pygame.K_SPACE and not game_active:
						game_active = True
						pipe_list.clear()
						bird_rect.center = (100,512)
						bird_movement = 0
						score = 0

				if event.type == GESTURE_FLAP and game_active:
					bird_movement = 0
					bird_movement -= FLAP_STRENGTH
					flap_sound.play()

				if event.type == GESTURE_FLAP and not game_active:
					game_active = True
					pipe_list.clear()
					bird_rect.center = (100,512)
					bird_movement = 0
					score = 0

				if event.type == SPAWNPIPE:
					pipe_list.extend(create_pipe())

				if event.type == BIRDFLAP:
					if bird_index < 2:
						bird_index += 1
					else:
						bird_index = 0
					bird_surface, bird_rect = bird_animation()

			screen.blit(bg_surface,(0,0))

			if game_active:
				bird_movement += gravity
				rotated_bird = rotate_bird(bird_surface)
				bird_rect.centery += bird_movement
				screen.blit(rotated_bird,bird_rect)
				game_active = check_collision(pipe_list)

				pipe_list = move_pipes(pipe_list)
				pipe_list = remove_pipes(pipe_list)
				draw_pipes(pipe_list)

				score += 0.01
				score_display('main_game')

				score_sound_countdown -= 1
				if score_sound_countdown <= 0:
					score_sound.play()
					score_sound_countdown = 100
			else:
				screen.blit(game_over_surface,game_over_rect)
				high_score = update_score(score,high_score)
				score_display('game_over')

			floor_x_pos -= FLOOR_SPEED
			draw_floor()
			if floor_x_pos <= -576:
				floor_x_pos = 0

			pygame.display.update()
			clock.tick(GAME_FPS)

	except KeyboardInterrupt:
		gesture_engine.stop()
	finally:
		pygame.quit()


if __name__ == "__main__":
	run_flappy_game()
