import numpy as np
import random

def knapsack_aco(values, weights, capacity, num_ants=10, max_iterations=100, evaporation_rate=0.5, alpha=1, beta=1):
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



# generate an np.array with 300 values
print('Generating values and weights...')
random.seed(42)
values = np.random.randint(1, 100, 300)
weights = np.random.randint(1, 100, 300)
capacity = 200
best_solution, best_value = knapsack_aco(values=values, weights=weights, capacity=capacity, num_ants=20, max_iterations=1000, evaporation_rate=0.4, alpha=1, beta=1)
print('Best solution: {}'.format(best_solution))
print('Best value: {}'.format(best_value))