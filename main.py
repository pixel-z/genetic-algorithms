import numpy as np
import random 
import json
import os
import requests

POPULATION_SIZE = 10
VECTOR_SIZE = 11
MATING_POOL_SIZE = 8
PASSED_FROM_PARENTS = 8
KEY = 'qF9wD9XQoX5Pzca7gYOPAQxP6eUozTMGIelPiQVnV8XiqKeVow'

overfit_vector = [0.0, -1.45799022e-12, -2.28980078e-13,  4.62010753e-11, -1.75214813e-10, -1.83669770e-15,  8.52944060e-16,  2.29423303e-05, -2.04721003e-06, -1.59792834e-08,  9.98214034e-10]

first_parent = overfit_vector

TRAIN_RATIO = 1

API = 'http://10.4.21.156'
MAX_DEG = 11

def urljoin(root, path=''):
    if path: root = '/'.join([root.rstrip('/'), path.rstrip('/')])
    return root

def send_request(id, vector, path):
    api = urljoin(API, path)
    vector = json.dumps(vector)
    response = requests.post(api, data={'id':id, 'vector':vector}).text
    if "reported" in response:
        print(response)
        exit()

    return response

def get_errors(id, vector):
    """
    returns python array of length 2 
    (train error and validation error)
    """
    for i in vector: assert -10<=abs(i)<=10
    assert len(vector) == MAX_DEG

    return json.loads(send_request(id, vector, 'geterrors'))

def submit(id, vector):
    """
    used to make official submission of your weight vector
    returns string "successfully submitted" if properly submitted.
    """
    for i in vector: assert -10<=abs(i)<=10
    assert len(vector) == MAX_DEG
    return send_request(id, vector, 'submit')


def initial_population():
    first_population = []
    for i in range(POPULATION_SIZE):
        first_population.append(np.copy(first_parent))
    
    for i in range(POPULATION_SIZE):
        for j in range(VECTOR_SIZE):
            vary = 0
            mutation_prob = random.randint(0, 100)
            if mutation_prob < 60:
                vary = 1 + random.uniform(-0.05*j, 0.05*j)
                rem = overfit_vector[j]*vary

                if abs(rem) < 10:
                    first_population[i][j] = rem
                elif abs(first_population[i][j]) >= 10:
                    first_population[i][j] = random.uniform(-1,1)
    return first_population


def get_population_fitness(population):
    fit = np.empty((POPULATION_SIZE, 3))

    for i in range(POPULATION_SIZE):
        KEY, list(population[i])
        fit[i][0] = error[0]
        fit[i][1] = error[1]
        fit[i][2] = abs(error[0]*TRAIN_RATIO + error[1]) 

    ret = np.column_stack((population, fit))
    ret = ret[np.argsort(ret[:,-1])]
    return ret


def mutation(vector):
    for i in range(VECTOR_SIZE):
        mutation_prob = random.randint(0, 100)
        if mutation_prob < 50:
            var = 1 + random.uniform(-0.3, 0.3)
            rem = overfit_vector[i]*var # multiply to create variations
            if abs(rem) <= 10:
                vector[i] = rem
    return vector


def crossover(parent1, parent2):
    child1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    child2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    r1 = random.random() 
    n = 3
        
    beta = (2 * r1)**((n + 1)**-1)
    if (r1 >= 0.5):
        beta = ((2*(1-r1))**-1)**((n + 1)**-1)

    p1 = np.array(parent1)
    p2 = np.array(parent2)
    child1 = 0.5*((1 + beta) * p1 + (1 - beta) * p2)
    child2 = 0.5*((1 - beta) * p1 + (1 + beta) * p2)

    return [child1, child2]


def create_children(mating_pool):
    mating_pool = mating_pool[:, :-3]
    children = []
    for i in range( int(POPULATION_SIZE/2)):
        index1 = random.randint(0, MATING_POOL_SIZE-1)
        index2 = random.randint(0, MATING_POOL_SIZE-1)

        parent1 = mating_pool[index1]
        parent2 = mating_pool[index2]

        offsprings = crossover(parent1, parent2)

        child1, child2 = offsprings[0], offsprings[1]
        
        child1 = mutation(child1)
        child2 = mutation(child2)

        children.append(child1)
        children.append(child2)

    return children


def create_next_gen(parents_fitness, children):
    child_fitness = get_population_fitness(children)[:(POPULATION_SIZE-PASSED_FROM_PARENTS)]
    parents_fitness = parents_fitness[:PASSED_FROM_PARENTS]
    generation = np.concatenate((parents_fitness, child_fitness))
    generation = generation[np.argsort(generation[:,-1])]
    return generation


no_gen = 10
population = []
population_fitness = []
vectors = []

population = initial_population()
population_fitness = get_population_fitness(population)

for generation in range(no_gen):   

    fitness_population = fitness_population[np.argsort(fitness_population[:,-1])]
    mating_pool = fitness_population[:MATING_POOL_SIZE]
    children = create_children(mating_pool)
    population_fitness = create_next_gen(mating_pool, children)

    population = population_fitness[:, :-3]

    population_vectors = []
    for vector in population:
        population_vectors.append(vector.toList())

    vectors.append(population_vectors)

with open('output.json','w') as f: 
    json.dump(vectors, f, indent = 2) 