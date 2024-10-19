import numpy as np
import matplotlib.pyplot as plt
import random

random.seed(1)


def get_columns(data, index):
    result = [row[index] for row in data]
    return result


def load_data_from_file(filename="advertising.csv"):
    data = np.genfromtxt(filename, dtype=None, delimiter=',', skip_header=1)

    tv_data = get_columns(data, 0)
    radio_data = get_columns(data, 1)
    newspaper_data = get_columns(data, 2)
    sale_data = get_columns(data, 3)

    features_x = np.array([[1, x1, x2, x3]
                           for x1, x2, x3 in zip(tv_data, radio_data, newspaper_data)])
    sales_y = np.array(sale_data)

    return features_x, sales_y


features_x, sales_y = load_data_from_file(filename="advertising.csv")


def generate_random_value(bound=10):
    return (random.random()*2 - 1)*bound


def create_individual(n=4, bound=10):
    individual = [generate_random_value(bound) for _ in range(n)]
    return individual


def compute_loss(individual):
    theta = np.array(individual)
    y_hat = features_x.dot(theta)
    loss = np.multiply((y_hat-sales_y), (y_hat-sales_y)).mean()

    return loss


def compute_fitness(individual):
    loss = compute_loss(individual)
    fitness_value = 1/(loss+1)

    return fitness_value


def crossover(individual_1, individual_2, crossover_rate=0.9):
    individual_1_new = individual_1.copy()
    individual_2_new = individual_2.copy()

    for i in range(len(individual_1)):
        if random.random() < crossover_rate:
            individual_1_new[i] = individual_2[i]
            individual_2_new[i] = individual_1[i]

    return individual_1_new, individual_2_new


def mutate(individual, mutation_rate=2.0):
    individual_m = individual.copy()

    for i in range(len(individual_m)):
        if random.random() < mutation_rate:
            individual_m[i] = generate_random_value()
    return individual_m


def initialize_population(m):
    population = [create_individual() for _ in range(m)]
    return population


def selection(sorted_old_population, m=100):
    index1 = random.randint(0, m-1)
    while True:
        index2 = random.randint(0, m-1)
        if (index2 != index1):
            break

    individual_s = sorted_old_population[index1]

    if index2 > index1:
        individual_s = sorted_old_population[index2]

    return individual_s


def create_new_population(old_population, elitism=2, gen=1):
    m = len(old_population)
    sorted_population = sorted(old_population, key=compute_fitness)

    if gen % 100 == 0:
        print("best loss: ", compute_loss(
            sorted_population[m-1]), "with chromsome: ", sorted_population[m-1])

    new_population = []
    while len(new_population) < m-elitism:
        # Selection
        individual_s1 = selection(sorted_population)
        individual_s2 = selection(sorted_population)

        # crossover
        individual_c1, individual_c2 = crossover(individual_s1, individual_s2)

        # mutation
        individual_m1 = mutate(individual_c1)
        individual_m2 = mutate(individual_c2)

        new_population.append(individual_m1)
        new_population.append(individual_m2)

    # Copy elitism chromosomes that have best fitness score to the next generation
    for ind in sorted_population[m-elitism:]:
        new_population.append(ind.copy())

    return new_population, compute_loss(sorted_population[m-1])


def run_GA():
    n_generations = 100
    m = 600
    population = initialize_population(m)

    losses_list = []
    for i in range(n_generations):
        population, loss = create_new_population(
            old_population=population, elitism=2, gen=i)

        losses_list.append(loss)
    return losses_list


def visualize_loss(losses_list):
    plt.plot(losses_list, c='red')
    plt.xlabel('Generation')
    plt.ylabel('losses')
    plt.show()


losses_list = run_GA()
visualize_loss(losses_list)
