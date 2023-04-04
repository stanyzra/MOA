"""
Alunos:
Stany Helberth de Souza Gomes da Silva - RA112683
Gabriel de Souza Vendrame - RA112681
Henrique Ribeiro Favaro - 115408
"""

import numpy as np
import random

random.seed(42)
initialValues = []
initialWeights = []
for i in range(300):
    initialValues.append(random.randint(1, 100))
    initialWeights.append(random.randint(1, 100))

def knapsack_aco(values, weights, capacity, num_ants=10, max_iterations=100, evaporation_rate=0.5, alpha=1, beta=1):
    """
    Solves the knapsack problem using the Ant Colony Optimization algorithm.

    Parameters
    ----------
    values : NP.array-like
        The values of the items.
    weights : NP.array-like
        The weights of the items.
    capacity : int
        The capacity of the knapsack.
    num_ants : int, optional
        The number of ants to use. The default is 10.
    max_iterations : int, optional
        The maximum number of iterations. The default is 100.
    evaporation_rate : float, optional
        The evaporation rate of the pheromone trail. The default is 0.5.
    alpha : float, optional
        The alpha parameter of the ACO algorithm. The default is 1.
    beta : float, optional
        The beta parameter of the ACO algorithm. The default is 1.
    """
    num_items = len(values)

    # Initialize pheromone trail and heuristic information
    pheromone = np.ones(num_items) / num_items
    heuristic = values / weights

    # Initialize best solution and its value
    best_solution = None
    best_value = 0

    for iteration in range(max_iterations):
        print("Running iteration: ", iteration)
        # Initialize ants and their solutions
        ants = np.zeros((num_ants, num_items))
        ant_values = np.zeros(num_ants)
        for ant in range(num_ants):
            available_items = list(range(num_items))
            available_capacity = capacity
            while available_items and available_capacity > 0:
                # Select the next item based on the pheromone trail and heuristic information
                prob = pheromone[available_items] ** alpha * heuristic[available_items] ** beta
                prob /= np.sum(prob)
                next_item = np.random.choice(available_items, p=prob)
                # Add the item to the ant's solution if it fits in the knapsack
                if weights[next_item] <= available_capacity:
                    ants[ant, next_item] = 1
                    ant_values[ant] += values[next_item]
                    available_capacity -= weights[next_item]
                available_items.remove(next_item)
        
        # Update the best solution and its value
        best_ant_index = np.argmax(ant_values)
        if ant_values[best_ant_index] > best_value:
            best_solution = ants[best_ant_index]
            best_value = ant_values[best_ant_index]

        # Update the pheromone trail
        delta_pheromone = np.zeros(num_items)
        for ant in range(num_ants):
            delta_pheromone += ants[ant] * ant_values[ant] / capacity
        pheromone = (1 - evaporation_rate) * pheromone + evaporation_rate * delta_pheromone

    return best_solution, best_value

# values = np.random.randint(1, 100, 300)
values = np.array(initialValues)
# weights = np.random.randint(1, 100, 300)
weights = np.array(initialWeights)
capacity = 370

best_solution, best_value = knapsack_aco(
    values=values, 
    weights=weights, 
    capacity=capacity, 
    num_ants=30, 
    max_iterations=300, 
    evaporation_rate=0.5, 
    alpha=0.3, 
    beta=1
)

used_itens = []
for i in range(len(best_solution)):
    if best_solution[i] == 1:
        used_itens.append(i)

print('Melhor arranjo de itens: {}'.format(used_itens))
print('Melhor valor alcançado: {}'.format(best_value))
print('Peso total utilizado: {}'.format(np.sum(weights[used_itens])))
print('Critério de parada utilizado: Máximo de 300 iterações')