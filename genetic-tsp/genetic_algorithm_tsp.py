import numpy as np
import numpy.typing as npt
import math
import copy


class City:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.coords = (x, y)


class Individual:
    def __init__(self, chromosome: npt.ArrayLike):
        self.chromosome = np.array(chromosome)
        self.fitness = 0
        self.normal_fitness = 0
        self.probability = 0

    def __str__(self):
        return f"Chromosome: {self.chromosome}, Fitness: {self.fitness}, Normal Fitness: {self.normal_fitness}, Probability: {self.probability}"


def fitness_function(individual: Individual, cities: list[City]) -> float:
    fitness = 0
    for i in range(len(individual.chromosome)):
        operation = i + 1
        if i == len(individual.chromosome) - 1:
            operation = 0

        city1 = cities[individual.chromosome[i]]
        city2 = cities[individual.chromosome[operation]]
        fitness += math.dist(city1.coords, city2.coords)
    return 1 / (fitness), fitness


def generate_population(population_size: int, chromosome_size: int) -> npt.ArrayLike:
    population = []
    for _ in range(population_size):
        chromosome = np.random.permutation(chromosome_size)
        chromosome = Individual(chromosome)
        population.append(chromosome)
    return np.array(population)


def generate_cities(number_of_cities: int) -> npt.ArrayLike:
    cities = []
    for _ in range(number_of_cities):
        cities.append(City(x=np.random.randint(0, 5), y=np.random.randint(0, 5)))
    return np.array(cities)


def calculate_probabilities(population: npt.ArrayLike) -> npt.ArrayLike:
    total_fitness = np.sum(np.array([individual.fitness for individual in population]))
    for individual in population:
        individual.probability = individual.fitness / total_fitness
    return population


def select_best_parents(
    population: npt.ArrayLike, best_parents_size: int
) -> npt.ArrayLike:
    parents = np.zeros(best_parents_size, dtype=Individual)
    population = sorted(
        population, key=lambda individual: individual.fitness, reverse=True
    )
    for i in range(best_parents_size):
        parents[i] = population[i]

    return parents, np.delete(population, np.s_[0:best_parents_size], axis=0)


def roulette_wheel_selection(population: npt.ArrayLike) -> npt.ArrayLike:
    r = np.random.uniform(0, 1)
    acc = 0
    for i, individual in enumerate(population):
        acc += individual.probability
        if acc > r:
            return individual


def mutation(individual: Individual, mutation_rate: float) -> Individual:
    if np.random.uniform(0, 1) < mutation_rate:
        i = np.random.randint(0, len(individual.chromosome))
        j = np.random.randint(0, len(individual.chromosome))

        while i == j:
            j = np.random.randint(0, len(individual.chromosome))

        individual.chromosome[i], individual.chromosome[j] = (
            individual.chromosome[j],
            individual.chromosome[i],
        )

    return individual


def two_OPT_mutation(individual: Individual, mutation_rate: float) -> Individual:
    if np.random.uniform(0, 1) < mutation_rate:
        i = np.random.randint(0, len(individual.chromosome))
        j = np.random.randint(0, len(individual.chromosome))

        while i == j:
            j = np.random.randint(0, len(individual.chromosome))

        if i > j:
            i, j = j, i

        individual.chromosome[i:j] = individual.chromosome[i:j][::-1]

    return individual


def reverse_path(path: list, i: int, j: int) -> list:
    for k in range((j+1-i)//2):
        path[j-k], path[i+k] = path[i+k], path[j-k]
    return path

def best_improvement(individual: Individual, cities: list[City], best_fitness: float) -> Individual:
    individual_fitness = best_fitness
    best_i, best_j = 0, 0
    for i in range(len(individual.chromosome)):
        for j in range(i + 1, len(individual.chromosome)):
            dist = math.dist(cities[individual.chromosome[i-1]].coords, cities[individual.chromosome[i]].coords) + math.dist(cities[individual.chromosome[j]].coords, cities[individual.chromosome[(j+1)%len(individual.chromosome)]].coords)
            change_dist = math.dist(cities[individual.chromosome[i-1]].coords, cities[individual.chromosome[j]].coords) + math.dist(cities[individual.chromosome[i]].coords, cities[individual.chromosome[(j+1)%len(individual.chromosome)]].coords)
            fitness = individual_fitness - dist + change_dist
            if fitness < best_fitness:
                best_i, best_j = i, j
                best_fitness = fitness
    individual.chromosome = reverse_path(individual.chromosome, best_i, best_j)
    return individual


def OX_crossover(parents: npt.ArrayLike) -> npt.ArrayLike:

    offspring = np.zeros((2, len(parents[0].chromosome)), dtype=Individual)

    slice_1 = np.random.randint(0, len(parents[0].chromosome))
    slice_2 = np.random.randint(0, len(parents[1].chromosome))

    while slice_1 == slice_2:
        slice_2 = np.random.randint(0, len(parents[1].chromosome))

    if slice_1 > slice_2:
        slice_1, slice_2 = slice_2, slice_1

    offspring[0] = copy.deepcopy(parents[0].chromosome)
    offspring[1] = copy.deepcopy(parents[1].chromosome)

    visited_genes_1 = np.array(len(parents[0].chromosome) * [False])
    visited_genes_2 = np.array(len(parents[1].chromosome) * [False])

    visited_genes_1[parents[0].chromosome[slice_1:slice_2].astype(np.int64)] = True
    visited_genes_2[parents[1].chromosome[slice_1:slice_2].astype(np.int64)] = True

    j = slice_2
    for i in range(len(parents[0].chromosome)):
        if not visited_genes_1[parents[1].chromosome[i]]:
            offspring[0][j] = parents[1].chromosome[i]
            visited_genes_1[parents[1].chromosome[i]] = True
            j = (j + 1) % len(parents[0].chromosome)

    j = slice_2
    for i in range(len(parents[1].chromosome)):
        if not visited_genes_2[parents[0].chromosome[i]]:
            offspring[1][j] = parents[0].chromosome[i]
            visited_genes_2[parents[0].chromosome[i]] = True
            j = (j + 1) % len(parents[1].chromosome)

    return np.array([Individual(offspring[0]), Individual(offspring[1])])


def population_control(
    population: npt.ArrayLike, offspring: npt.ArrayLike
) -> npt.ArrayLike:
    population = sorted(population, key=lambda individual: individual.normal_fitness)
    offspring = sorted(offspring, key=lambda individual: individual.normal_fitness)

    new_pop = []

    i, j = 0, 0

    for k in range(len(population)):
        if i == len(population) or j == len(offspring): break # acabou uma das populaçoes

        if population[i].normal_fitness < offspring[j].normal_fitness:
            new_pop.append(population[i])
            i += 1
        else:
            new_pop.append(offspring[j])
            j += 1

        # ignora soluções parecidas
        while i < len(population) and new_pop[-1].normal_fitness == population[i].normal_fitness:
            i += 1
        while j < len(offspring) and new_pop[-1].normal_fitness == offspring[j].normal_fitness:
            j += 1
    
    if len(new_pop) < len(population): # quando acabar uma das duas populações
        while i < len(population) and len(new_pop) < len(population):
            new_pop.append(population[i])
            i += 1
        while j < len(offspring) and len(new_pop) < len(offspring):
            new_pop.append(offspring[j])
            j += 1
        while len(new_pop) < len(population):  # acabou as duas populações
            new_pop.append(Individual(np.random.permutation(len(population[0].chromossome))))

    return new_pop


def main():
    """
    9042 gerações
    39 indivíduos
    0.04 mutação

    40290.56369780305
    """
    mutation_rate = 0.04
    num_generations = 2000

    gen = 0

    cities = []
    with open("./genetic-tsp/att48.txt", "r") as input_file:
        for line in input_file.readlines():
            line = line.split()
            cities.append(City(x=int(line[1]), y=int(line[2])))

    parents = generate_population(38, len(cities))

    current_fitness = -1

    count = 0

    while count < num_generations:
        for i, individual in enumerate(parents):
            individual.fitness, individual.normal_fitness = fitness_function(
                individual, cities
            )

        parents = calculate_probabilities(parents)

        parents = sorted(parents, key=lambda individual: individual.normal_fitness)

        offspring = np.zeros(len(parents), dtype=Individual)

        for i in range(0, len(parents), 2):
            father = roulette_wheel_selection(parents)
            mother = roulette_wheel_selection(parents)

            offspring[i : i + 2] = OX_crossover(np.array([father, mother]))

            offspring[i].fitness, offspring[i].normal_fitness = fitness_function(
                offspring[i], cities
            )

            (
                offspring[i + 1].fitness,
                offspring[i + 1].normal_fitness,
            ) = fitness_function(offspring[i + 1], cities)

            offspring[i] = best_improvement(offspring[i], cities, offspring[i].fitness)
            offspring[i + 1] = best_improvement(offspring[i + 1], cities, offspring[i+1].fitness)
        offspring = sorted(
            offspring, key=lambda individual: individual.normal_fitness, reverse=True
        )

        parents = population_control(parents, offspring)
        print(f"Population")
        acc = 0
    

        for i in range(1, len(parents)):
            if parents[i].normal_fitness == parents[i - 1].normal_fitness:
                acc += 1
        print(f"Acc: {acc} at generation {gen}.")
        print(f"Best fitness: {parents[0].normal_fitness} at generation {gen}.")

        if math.floor(parents[0].normal_fitness) != math.floor(current_fitness):
            count = 0
            current_fitness = parents[0].normal_fitness
        else:
            count += 1

        for i, individual in enumerate(parents):
            if i == 0:
                continue
            individual = two_OPT_mutation(individual, mutation_rate)

        gen += 1

    print(f"Best fitness: {parents[0].normal_fitness} at generation {gen}.")
    print(f"Best solution: ")
    print(parents[0].chromosome)
    print(f"at generation {gen}.")



if __name__ == "__main__":
    main()
