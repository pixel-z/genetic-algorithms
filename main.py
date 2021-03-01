import numpy as np
import json
import requests
import random

# Error on overfit vector => [13510723304.19212, 368296592820.6967]

API_KEY = 'qF9wD9XQoX5Pzca7gYOPAQxP6eUozTMGIelPiQVnV8XiqKeVow'
API_ENDPOINT = 'http://10.4.21.156'
MAX_DEG = 11
TRAINING_FACTOR = 0.7
MAX_PARENTS_SIZE = 

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


# def random_weight():
#     return random.uniform(-10, 10)


def mutate_vector(vector):
    for i in range(len(vector)):
        mutation_probability = random.randint(0, 10)
        if mutation_probability < 4:
            multiplication_factor = 1 + random.uniform(-0.05, 0.05)
            new_value = vector[i] * multiplication_factor
            if new_value >= -10 and new_value <= 10:
                vector[i] = new_value
    
    return vector


def create_generation_zero(starting_vector):
    population = []
    for i in range(10):
        population.append(mutate_vector(starting_vector))
    return population


def fitness_measure(errors):
    global TRAINING_FACTOR
    training_error, validation_error = errors
    return training_error * TRAINING_FACTOR + validation_error


def get_mating_indexes(fitness):
    population_fitness = population
    
    
if __name__ == "__main__":
    overfit_vector = get_overfit_vector()
    print(overfit_vector)
    # print(get_errors(overfit_vector))
    generations = [create_generation_zero(overfit_vector)]
    errors = []
    
    for generation_number in range(10):
        generation = generations[generation_number]
        member_errors = [get_errors(member) for member in generation]
        errors.append(member_errors)
        fitness_values = [fitness_measure(member_error) for member_error in member_errors]
        mating_pool_indexes = get_mating_indexes(fitness_values)


