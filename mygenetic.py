import random

import numpy as np
import mybird

population_size = 100
parents_to_keep = 1
revive = 20
soft_mut_prob = 0.4
hard_mut_prob = 0.8
dummy_bird = mybird.Passaro("dummy")
start_end = dummy_bird.start_end
print(start_end)

def mutation_func(offspring):
	for chromosome_idx in range(offspring.shape[0]):
		random_split_point = random.choice(start_end)
		
		for gene_idx in range(random_split_point[0], random_split_point[1]):
			if np.random.random() > soft_mut_prob:
				offspring[chromosome_idx, gene_idx] += (0.5) * np.random.random()
			else:
				if np.random.random() > hard_mut_prob:
					offspring[chromosome_idx, gene_idx] *= -1
				
	return offspring

def crossover_func(parents, offspring_size):
	offspring = []
	idx = 0
#	best_parents = parents[0:10, :].copy()
	while len(offspring) != offspring_size:
		parent1_idx = np.random.randint(len(parents))
		parent2_idx = np.random.randint(len(parents))

		parent1 = parents[parent1_idx].copy()
		parent2 = parents[parent2_idx].copy()

		# troca 1 camada da rede neural
#		random_split_point = random.choice(start_end)
#		parent1[random_split_point[0]:random_split_point[1]] = parent2[random_split_point[0]:random_split_point[1]]

		# troca 1 neuronio de 1 camada
		n = random.choice(range(parents.shape[1]))
		parent1[n] = parent2[n].copy()
		
		offspring.append(parent1)

		idx += 1

	return np.array(offspring)

def parent_selection_func(population, fitness, num_parents, best_brains):
	fitness_sorted = np.argsort(fitness * -1)

	parents = np.empty((num_parents, population.shape[1]))
	
	# melhores da geração
	for parent_num in range(parents_to_keep):
		parents[parent_num, :] = population[fitness_sorted[parent_num], :].copy()
	
	# melhores global
	if len(best_brains) > revive:
		bests = 2
		best_fitness = []
		for best in range(bests):
#			random_best = np.random.choice(range(len(best_brains)))
			parents[parents_to_keep + best, :] = best_brains[best][0].copy()
			best_fitness.append(best_brains[best][1].copy())
	else:
		bests = 0
		best_fitness = []
	
	# aleatorios
	fitness_random = []
	for parent_num in range(parents_to_keep+bests, num_parents):
		random_parent = np.random.choice(range(population.shape[0]))
		fitness_random.append(random_parent)
		parents[parent_num, :] = population[random_parent, :].copy()
	
	fitness_list = list(fitness[fitness_sorted[:parents_to_keep]]) + best_fitness + list([fitness[i] for i in fitness_random])
	return parents, fitness_list

def train_gen(bird_family, best_brains, num_parents):
	fitness = np.array([bird.score for bird in bird_family])
	population = np.array([bird.weights_vector() for bird in bird_family])
	fitness_sorted = np.argsort(fitness * -1)
	
	best_brains = [(best, best_fitness * 0.9) for best, best_fitness in best_brains]

	best = population[fitness_sorted[0]]
	best_fitness = fitness[fitness_sorted[0]]
	
	best_brains.append((best, best_fitness))
	if len(best_brains) > 5:
		best_brains.sort(key=lambda tuplas: tuplas[1])
		best_brains.reverse()
		best_brains.pop(-1)
		
	parents, fitness_list = parent_selection_func(population, fitness, num_parents, best_brains)

	offspring_size = len(fitness) - len(parents)
	offspring = crossover_func(parents, offspring_size)

	new_population = np.array(np.concatenate([parents, offspring], axis = 0))
	return new_population, best_brains