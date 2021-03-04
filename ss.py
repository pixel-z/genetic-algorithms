from client_moodle import get_errors, submit
import numpy as np
import random 
import json
import os

SECRET_KEY = 'dnLVLTHPAUOT2R1Ruj1sQvXxWBZZchp8u4WkyZGzaeTQCpyFXC'
POPULATION_SIZE = 30
VECTOR_SIZE = 11
MATING_POOL_SIZE = 10
FROM_PARENTS = 8
FILE_NAME_READ = 'team_5.json'
FILE_NAME_WRITE = 'team_5.json'
overfit_vector = [0.0, 0.1240317450077846, -6.211941063144333, 0.04933903144709126, 0.03810848157715883, 8.132366097133624e-05, -6.018769160916912e-05, -1.251585565299179e-07, 3.484096383229681e-08, 4.1614924993407104e-11, -6.732420176902565e-12]

first_parent = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

TRAIN_FACTOR = 1
fieldNames = ['Generation','Vector','Train Error','Validation Error', 'Fitness']

def where_json(fileName):
    return os.path.exists(fileName)

def write_json(data, filename = FILE_NAME_WRITE): 
    with open(filename,'w') as f: 
        json.dump(data, f, indent = 4) 

def initial_population():
    first_population = [np.copy(first_parent) for i in range(POPULATION_SIZE)]
    
    for i in range(POPULATION_SIZE):
        for index in range(VECTOR_SIZE):
            vary = 0
            mutation_prob = random.randint(0, 10)
            if mutation_prob < 3:
                if index <= 4:
                    vary = 1 + random.uniform(-0.05, 0.05)
                else:
                    vary = random.uniform(0, 1)
                rem = overfit_vector[index]*vary

                if abs(rem) < 10:
                    first_population[i][index] = rem
                elif abs(first_population[i][index]) >= 10:
                    first_population[i][index] = random.uniform(-1,1)

    return first_population

def calculate_fitness(population):
    fitness = np.empty((POPULATION_SIZE, 3))

    for i in range(POPULATION_SIZE):
        # error = get_errors(SECRET_KEY, list(population[i]))
        error = [1,1]
        fitness[i][0] = error[0]
        fitness[i][1] = error[1]
        fitness[i][2] = abs(error[0]*TRAIN_FACTOR + error[1]) 

    pop_fit = np.column_stack((population, fitness))
    pop_fit = pop_fit[np.argsort(pop_fit[:,-1])]
    return pop_fit

def create_mating_pool(population_fitness):
    population_fitness = population_fitness[np.argsort(population_fitness[:,-1])]
    mating_pool = population_fitness[:MATING_POOL_SIZE]
    return mating_pool

def mutation(child):

    for i in range(VECTOR_SIZE):
        mutation_prob = random.randint(0, 10)
        if mutation_prob < 3:
            if i <= 4:
                vary = 1 + random.uniform(-0.05, 0.05)
            else:
                vary = random.uniform(0, 1) # This was set to 1 + random.uniform(-0.05, 0.05) for trace
            rem = overfit_vector[i]*vary
            if abs(rem) <= 10:
                child[i] = rem
    return child
        

def crossover(parent1, parent2):

    child1 = np.empty(11)
    child2 = np.empty(11)

    u = random.random() 
    n_c = 3
        
    if (u < 0.5):
        beta = (2 * u)**((n_c + 1)**-1)
    else:
        beta = ((2*(1-u))**-1)**((n_c + 1)**-1)


    parent1 = np.array(parent1)
    parent2 = np.array(parent2)
    child1 = 0.5*((1 + beta) * parent1 + (1 - beta) * parent2)
    child2 = 0.5*((1 - beta) * parent1 + (1 + beta) * parent2)

    return child1, child2

def create_children(mating_pool):
    mating_pool = mating_pool[:, :-3]
    children = []
    for i in range( int(POPULATION_SIZE/2)):
        parent1 = mating_pool[random.randint(0, MATING_POOL_SIZE-1)]
        parent2 = mating_pool[random.randint(0, MATING_POOL_SIZE-1)]
        child1, child2 = crossover(parent1, parent2)
        
        child1 = mutation(child1)
        child2 = mutation(child2)

        children.append(child1)
        children.append(child2)


    return children  


def new_generation(parents_fitness, children):
    children_fitness = calculate_fitness(children)
    parents_fitness = parents_fitness[:FROM_PARENTS]
    children_fitness = children_fitness[:(POPULATION_SIZE-FROM_PARENTS)]
    generation = np.concatenate((parents_fitness, children_fitness))
    generation = generation[np.argsort(generation[:,-1])]
    return generation



def main():

    num_generations = 10
    population = []
    population_fitness = []

    if where_json(FILE_NAME_READ):
        with open(FILE_NAME_READ) as json_file:
            data = json.load(json_file)
            population = [dict_item["Vector"] for dict_item in data["Storage"][-POPULATION_SIZE:]]
            train = [dict_item["Train Error"] for dict_item in data["Storage"][-POPULATION_SIZE:]]
            valid = [dict_item["Validation Error"] for dict_item in data["Storage"][-POPULATION_SIZE:]]
            offset = [dict_item["Generation"] for dict_item in data["Storage"][-1:]]
            fitness = [abs(TRAIN_FACTOR*train[i] + valid[i]) for i in range(POPULATION_SIZE)]
            population_fitness = np.column_stack((population, train, valid, fitness))
            population_fitness = population_fitness[np.argsort(population_fitness[:,-1])]
    else:
        population = initial_population()
        population_fitness = calculate_fitness(population)
        data = {"Storage": []}
        with open(FILE_NAME_WRITE, 'w') as writeObj:
            json.dump(data, writeObj)

    for generation in range(num_generations):   

        mating_pool = create_mating_pool(population_fitness)
        children = create_children(mating_pool)
        population_fitness = new_generation(mating_pool, children)

        fitness = population_fitness[:, -3:] 
        population = population_fitness[:, :-3]      
        
        for i in range(POPULATION_SIZE):
            # submit_status = submit(SECRET_KEY, population[i].tolist())
            # assert "submitted" in submit_status
            with open(FILE_NAME_WRITE) as json_file:
                data = json.load(json_file)
                temporary = data["Storage"]
                rowDict = { 
                            "Generation": generation + 1,
                            "Vector": population[i].tolist(), 
                            "Train Error": fitness[i][0], 
                            "Validation Error": fitness[i][1],
                            "Fitness": fitness[i][2]}
                temporary.append(rowDict)
            write_json(data)

if __name__ == '__main__':
    main() 