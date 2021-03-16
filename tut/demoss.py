import numpy as np
import random 
import json
import os
import requests

SECRET_KEY = 'qF9wD9XQoX5Pzca7gYOPAQxP6eUozTMGIelPiQVnV8XiqKeVow'
POPULATION_SIZE = 10
VECTOR_SIZE = 11
MATING_POOL_SIZE = 8
PASSED_FROM_PARENTS = 8
FILE_NAME_READ =   './generations/391_400.json'
FILE_NAME_WRITE =  './generations/391_400.json'
TRACE_FILE = './generations/trace_391_400.json'
overfit_vector = [0.0, -1.45799022e-12, -2.28980078e-13,  4.62010753e-11, -1.75214813e-10, -1.83669770e-15,  8.52944060e-16,  2.29423303e-05, -2.04721003e-06, -1.59792834e-08,  9.98214034e-10]

first_parent = overfit_vector

TRAIN_FACTOR = 1
fieldNames = ['Generation','Vector','Train Error','Validation Error', 'Fitness']

trace = {"Trace": []}

API_ENDPOINT = 'http://10.4.21.156'
PORT = 3000
MAX_DEG = 11

#### utility functions
def urljoin(root, path=''):
    if path: root = '/'.join([root.rstrip('/'), path.rstrip('/')])
    return root

def send_request(id, vector, path):
    api = urljoin(API_ENDPOINT, path)
    vector = json.dumps(vector)
    response = requests.post(api, data={'id':id, 'vector':vector}).text
    if "reported" in response:
        print(response)
        exit()

    return response


#### functions that you can call
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


def where_json(fileName):
    return os.path.exists(fileName)


def write_json(data, filename = FILE_NAME_WRITE): 
    with open(filename,'w') as f: 
        json.dump(data, f, indent = 4) 


def initial_population():
    first_population = []
    for i in range(POPULATION_SIZE):
        first_population.append(np.copy(first_parent))
    
    for i in range(POPULATION_SIZE):
        for j in range(VECTOR_SIZE):
            vary = 0
            mutation_prob = random.randint(0, 100)
            if mutation_prob < 60:
                vary = 1 + random.uniform(-0.3, 0.3)
                rem = overfit_vector[j]*vary

                if abs(rem) < 10:
                    first_population[i][j] = rem
                elif abs(first_population[i][j]) >= 10:
                    first_population[i][j] = random.uniform(-1,1)
    return first_population


def get_population_fitness(population):
    fitness = np.empty((POPULATION_SIZE, 3))

    for i in range(POPULATION_SIZE):
        error = get_errors(SECRET_KEY, list(population[i]))
        # error = [1,1]
        fitness[i][0] = error[0]
        fitness[i][1] = error[1]
        fitness[i][2] = abs(error[0]*TRAIN_FACTOR + error[1]) 

    pop_fit = np.column_stack((population, fitness))
    pop_fit = pop_fit[np.argsort(pop_fit[:,-1])]
    return pop_fit


def create_mating_pool(fitness_population):
    fitness_population = fitness_population[np.argsort(fitness_population[:,-1])]
    mating_pool = fitness_population[:MATING_POOL_SIZE]
    return mating_pool


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
    details = []
    for i in range( int(POPULATION_SIZE/2)):
        detail1 = {}
        detail2 = {}

        parent1 = mating_pool[random.randint(0, MATING_POOL_SIZE-1)]
        parent2 = mating_pool[random.randint(0, MATING_POOL_SIZE-1)]

        detail1["Parent 1"] = parent1
        detail1["Parent 2"] = parent2

        detail2["Parent 1"] = parent1
        detail2["Parent 2"] = parent2

        child1, child2 = crossover(parent1, parent2)
        
        detail1["After Crossover"] = child1
        detail2["After Crossover"] = child2
        
        child1 = mutation(child1)
        child2 = mutation(child2)

        detail1["After Mutation"] = child1
        detail2["After Mutation"] = child2

        children.append(child1)
        children.append(child2)

        details.append(detail1)
        details.append(detail2)

    return children, details


def new_generation(parents_fitness, children):
    children_fitness = get_population_fitness(children)
    parents_fitness = parents_fitness[:PASSED_FROM_PARENTS]
    children_fitness = children_fitness[:(POPULATION_SIZE-PASSED_FROM_PARENTS)]
    generation = np.concatenate((parents_fitness, children_fitness))
    generation = generation[np.argsort(generation[:,-1])]
    return generation


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
    population_fitness = get_population_fitness(population)
    data = {"Storage": []}
    with open(FILE_NAME_WRITE, 'w') as writeObj:
        json.dump(data, writeObj)

for generation in range(num_generations):   

    this_generation = {"Generation": generation + 1}
    this_generation["Population"] = population_fitness[:, :-3].tolist()      
    this_generation["Details"] = []      

    mating_pool = create_mating_pool(population_fitness)
    children, details = create_children(mating_pool)
    population_fitness = new_generation(mating_pool, children)

    for childNumber in range(len(children)):
        childObject = {}
        childObject["Child Number"] = childNumber + 1
        childObject["Parent 2"] = details[childNumber]["Parent 2"].tolist()
        childObject["Parent 1"] = details[childNumber]["Parent 1"].tolist()

        childObject["After Crossover"] = details[childNumber]["After Crossover"].tolist()
        childObject["After Mutation"] = details[childNumber]["After Mutation"].tolist()

        this_generation["Details"].append(childObject)


    fitness = population_fitness[:, -3:] 
    population = population_fitness[:, :-3]

    
    for i in range(POPULATION_SIZE):
        # submit_status = submit(SECRET_KEY, population[i].tolist())
        # assert "submitted" in submit_status
        with open(FILE_NAME_WRITE) as json_file:
            # print("hello")
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
    
    trace["Trace"].append(this_generation)

write_json(trace, filename=TRACE_FILE)