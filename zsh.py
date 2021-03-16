import numpy as np
import json
import requests
import random

# Error on overfit vector => [13510723304.19212, 368296592820.6967]
# [0.0, -1.45799022e-12, -2.28980078e-13, 4.62010753e-11, -1.75214813e-10, -1.8366977e-15, 8.5294406e-16, 2.29423303e-05, -2.04721003e-06, -1.59792834e-08, 9.98214034e-10]
# SHRADHA -> [177822470183619.44, 335853778200767.6] 

API_KEY = 'qF9wD9XQoX5Pzca7gYOPAQxP6eUozTMGIelPiQVnV8XiqKeVow'
API_ENDPOINT = 'http://10.4.21.156'
MAX_DEG = 11
TRAINING_COEF = 0.7
MAX_PARENTS_SIZE = 10
GENERATIONS = 10
VECTOR_SIZE = 11

def urljoin(root, path=''):
    if path:
        root = '/'.join([root.rstrip('/'), path.rstrip('/')])
    return root

def send_request(vector, path):
    api = urljoin(API_ENDPOINT, path)
    vector = json.dumps(vector)
    response = requests.post(
        api,
        data={
            'id': API_KEY,
            'vector': vector
        }
    ).text
    if "reported" in response:
        print(response)
        exit()
    return response

def get_errors(vector):
    for i in vector:
        assert 0 <= abs(i) <= 10
    assert len(vector) == MAX_DEG

    return json.loads(send_request(vector, 'geterrors'))

def get_overfit_vector():
    with open("./overfit.txt", "r") as f:
        return json.load(f)

def mutate_vector(vector):
    for i in range(VECTOR_SIZE):
        mutation_prob = random.randint(0, 10)
        if mutation_prob < 5:
            vary = 1 + random.uniform(-0.3, 0.3)
            rem = overfit_vector[i]*vary
            if abs(rem) <= 10:
                child[i] = rem
    vector[VECTOR_SIZE-1] -= 0.1
    return vector

def crossover(parent1, parent2):
    pass

def create_generation_zero(starting_vector):
    population = []
    for i in range(10):
        population.append(mutate_vector(starting_vector))
    return population


def submit(id, vector):
    """
    used to make official submission of your weight vector
    returns string "successfully submitted" if properly submitted.
    """
    for i in vector: assert 0<=abs(i)<=10
    assert len(vector) == MAX_DEG
    return send_request(id, vector, 'submit')
    
if __name__ == "__main__":
    overfit_vector = get_overfit_vector()
    generations = [create_generation_zero(overfit_vector)]
    
    for gen_no in range(GENERATIONS):
        generation = generations[gen_no]

