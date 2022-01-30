import os
import sys
import pygame
import random
import numpy as np

import mybird
import mygenetic

class Passaro(mybird.Passaro):
	def __init__(self, name):
		super().__init__(name)
		
		self.sprite_path = random.sample([
			"./assets/sprites/yellowbird-midflap.png",
			"./assets/sprites/bluebird-midflap.png",
			"./assets/sprites/redbird-midflap.png"
		], 1)[0]
		self.image = pygame.image.load(self.sprite_path).convert()
		self.rect = self.image.get_rect(center = (50, screen_y / 2))
		
	def jump(self):
		self.passaro_mov = 0
		self.passaro_mov -= (4 * (120 / fps))
	
	def gravity(self):
		self.passaro_mov += (gravity * (120 / speed))
		
pygame.init()
pygame.font.init()

screen_x, screen_y = 288, 512
screen = pygame.display.set_mode((screen_x, screen_y))
clock = pygame.time.Clock()

bg = pygame.image.load("./assets/sprites/background-day.png").convert()
floor = pygame.image.load("./assets/sprites/base.png").convert()
pipe = pygame.image.load("./assets/sprites/pipe-green.png").convert()

flip_pipe = pygame.transform.flip(pipe, False, True)
pipe_list = []

text_font = pygame.font.SysFont('Comic Sans MS', 30)

# variaveis do jogo
gravity = 0.25
playing = True
time_step = 0
pipe_time = 0
fps = 60
speed = 60
generation = 0
global_score = 0

# [altura do passaro, distancia ao cano, altura cano de baixo]
#observation = [0.5, 0.82, 0.35, 0.2]

# [altura do passaro, altura cano de cima, altura cano de baixo]
dummy_cano = [(0.9, 0.5), (0.9, 0.25)]
best_brains = []

# eventos do jogo
#SPAWNPIPE = pygame.USEREVENT
#pygame.time.set_timer(SPAWNPIPE, int(900))
floor_x = 0

def draw_floor():
	screen.blit(floor, (floor_x, screen_y * 4/5))
	screen.blit(floor, (floor_x + screen_x, screen_y * 4/5))

def create_pipe():
#	random_h = random.randint(- int(screen_y / 8), int(screen_y / 8) )
	random_h = random.choice(np.linspace(- int(screen_y / 8), int(screen_y / 8), 5))
	bot_pipe = pipe.get_rect(midtop = (screen_x * 1.5, screen_y / 2 + random_h ) )
	
	top_pipe = pipe.get_rect(midbottom = (screen_x * 1.5, screen_y / 4 + random_h))
	
	return bot_pipe, top_pipe
	
def move_pipes(pipe_list):
	remove = False
	for pipe1 in pipe_list:
		pipe1.centerx -= 2 * (120 / speed)
		if pipe1.centerx < -20:
			pipe_list.remove(pipe1)
			remove = True
	return pipe_list, remove

def draw_pipes(pipe_list):
	for pipe1 in pipe_list:
		if pipe1.bottom >= screen_y:
			screen.blit(pipe, pipe1)
		else:
			screen.blit(flip_pipe, pipe1)
			

def check_collision(pipe_list):
	if (passaro_rect.top <= - 70) or (passaro_rect.bottom > screen_y * 4/5):
#			print("death, number:{}, score: {:.4f}".format(
#				passaro_obj.name, passaro_obj.score))
		passaro_obj.score -= 0.105
		passaro_obj.score += global_score
		return False
	
	for pipe1 in pipe_list:
		if passaro_rect.colliderect(pipe1):
#			print("death, number:{}, score: {:.4f}".format(
#				passaro_obj.name, passaro_obj.score)) 
			passaro_obj.score += global_score
			return False
	
	return True

def reset(bird_family, generation, best_brains):
	population = [bird.weights_vector() for bird in bird_family]
#	new_population, best_brains = myga.train_gen(population, fitness_function, best_brains)	
	
	num_parents = int(population_size * 0.3)
	new_population, best_brains = mygenetic.train_gen(bird_family, best_brains, num_parents)
	
	pop_score = np.array([bird.score for bird in bird_family])
#	print(np.array(sorted(pop_score)))
	print("#{})  min: {:.5f}  |  mean: {:.5f}  |  max: {:.5f}  | std: {:.5f}".format(
		str(generation).zfill(3),
		pop_score.min(), pop_score.mean(), pop_score.max(), pop_score.std()))

	new_bird_family = [Passaro("{}_{}".format(generation, str(i).zfill(2))
							  ).reborn(new_population[i]) if bird_family[i].score > 0.2 * pop_score.mean() else Passaro("{}_{}".format(generation, i)
							).reborn(best_brains[i % len(best_brains)][0]) for i in range(len(population))]
		
	return new_bird_family, best_brains

population_size = 100
bird_family = [Passaro("{}_{}".format(generation, i)) for i in range(population_size)]

save_path = os.listdir("./models")
print(len(save_path))
for i, save in enumerate(save_path):
	bird_family[i].brain.load_weights(filepath = "./models/" + save)

def fitness_function(solution, solution_idx):
	if type(solution_idx) == int:
		return (bird_family[solution_idx]).score
	else:
		print(solution_idx)
		pass

while True:
	screen.blit(bg, (0,-12))
	alive = len(bird_family)
	
	for passaro_obj in bird_family:
		alive -= not passaro_obj.vivo
		
	if alive > 0:
		pipe_list, remove = move_pipes(pipe_list)
		if remove:
			global_score += 0.5
		draw_pipes(pipe_list)
	
	if alive == 0:
		bird_family, best_brains = reset(bird_family, generation, best_brains)
		pipe_time = 0
		time_step = 0
		generation += 1
		global_score = 0
		pipe_list.clear()   
		
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			pygame.quit()  
			sys.exit()
		elif event.type == pygame.KEYDOWN:
			if event.key == pygame.K_ESCAPE:
				for bird in bird_family:
					bird_score_str = "{:.3f}".format(bird.score).replace(".", "F")
					bird.brain.save_weights(filepath = "./models/{}_{}.hdf5".format(bird_score_str, bird.name))
					
			if event.key == pygame.K_SPACE:
				save_path = os.listdir("./models")
				for i, save in enumerate(save_path):
					bird_family[i].brain.load_weights(filepath = "./models/" + save)
				
			if event.key == pygame.K_0:
				bird_family = [Passaro("{}_{}".format(generation, i)) for i in range(population_size)]
				
			if event.key == pygame.K_1:
				pop_score = np.array([bird.score for bird in bird_family])
				print(np.sort(pop_score))
				
			if event.key == pygame.K_2:
				speed = 30
				
			if event.key == pygame.K_3:
				speed = 60
	
	cano_position = []
	if len(pipe_list) > 1:
		for cano in pipe_list:
			if cano.center[0] < screen_x and cano.center[0] > 0:
				if cano.topleft[1] < 0:
					# cano de cima
					pipe_x, pipe_y = cano.bottomleft

				if cano.topleft[1] > 0:
					# cano de baixo
					pipe_x, pipe_y = cano.topleft

				cano_position.append((pipe_x / screen_x, pipe_y / screen_y))

	while len(cano_position) < 2:
		cano_position += dummy_cano
#		
	for passaro_obj in bird_family:
		if passaro_obj.vivo:
			passaro_image = passaro_obj.image
			passaro_rect = passaro_obj.rect

			passaro_obj.gravity()
			passaro_obj.update()

			if time_step % 5 == 0 and passaro_obj.vivo:
				time_step = 0
				time_step += 1

				observation = []
				try:
					bird_x, bird_y = passaro_rect.center
					
#					observation.append(bird_y / screen_y)
#					if (cano_position[1][0]) > 0.05:
					if True:
						observation.append(cano_position[1][0]) # distancia passaro cano
	#					observation.append(cano_position[0][1] - cano_position[1][1]) # gap entre os canos
						observation.append(bird_y / screen_y - cano_position[0][1]) # altura do cano de cima
						observation.append(bird_y / screen_y - cano_position[1][1]) # altura do cano de baixo

					else:
#						observation.append(cano_position[3][0] - bird_x / screen_x) # distancia passaro cano
	#					observation.append(cano_position[0][1] - cano_position[1][1]) # gap entre os canos
						observation.append(cano_position[2][1]) # altura do cano de cima
						observation.append(cano_position[3][1]) # altura do cano de baixo

					thinking = passaro_obj.think([observation])
					if thinking > 0.5:
						passaro_obj.jump()
						y = passaro_obj.update()
					else:
						pass

				except Exception as e:
					print(e)
			else:
				time_step += 1
				
			screen.blit(passaro_image, passaro_rect)
			passaro_obj.vivo = check_collision(pipe_list)
			passaro_obj.score += 0.005

	
	if len(pipe_list) < 2:
		pipe_list.extend(create_pipe())

	draw_floor()
	floor_x -= 0.5 * (120 / speed)
	if floor_x <= - screen_x:
		floor_x = 0
	
	
	text = text_font.render("alive: {}".format(str(int(alive)).zfill(3)), False, (255, 255, 255))
	screen.blit(text,(int(screen_x * 0.5),0))
	
	global_score_text = text_font.render("Score: {}".format(str(int(global_score)).zfill(3)), False, (36, 32, 18))
	screen.blit(global_score_text,(int(screen_x * 0.25), int(screen_y * 0.9)))

	pygame.display.update()
	clock.tick(fps)