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


def mutate_vector(vector, probability=4):
    return_vector = []
    for i in range(len(vector)):
        mutation_probability = random.randint(0, 10)
        # if mutation_probability < probability:
        if mutation_probability < probability:
            multiplication_factor = 1 + random.uniform(-0.5, 0.5)
            new_value = vector[i] * multiplication_factor
            if new_value >= -10 and new_value <= 10:
                return_vector.append(new_value)
            else:
                return_vector.append(vector[i])
        else:
            return_vector.append(vector[i])
    
    return return_vector


def create_generation_zero(starting_vector):
    population = []
    for i in range(10):
        population.append(mutate_vector(starting_vector, probability=8))
    return population


def fitness_measure(errors):
    global TRAINING_COEF
    training_error, validation_error = errors
    return training_error * TRAINING_COEF + validation_error


def get_mating_indexes(fitness):
    pass

    
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
    # print(overfit_vector)
    # print(get_errors(overfit_vector))
    generations = [create_generation_zero(overfit_vector)]
    # print(generations)
    errors = [get_errors(vec) for vec in generations[0]]
    
    for generation_number in range(10):
        generation = generations[generation_number]
        member_errors = [get_errors(member) for member in generation]
        errors.append(member_errors)
        fitness_values = [fitness_measure(member_error) for member_error in member_errors]
        mating_pool_indexes = get_mating_indexes(fitness_values)

    # print(submit(API_KEY, get_overfit_vector(API_KEY)))
