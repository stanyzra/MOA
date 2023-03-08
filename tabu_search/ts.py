import numpy as np


def fitness(solution, weights, values, capacity):
    max_value = 0
    max_weigth = 0
    # print(solution)
    for i, solution_value in np.ndenumerate(solution):
        # print(solution_value)
        if solution_value == 1:
            max_value += values[i]
            max_weigth += weights[i]
    # print(max_weigth)
    # print(capacity)
    if max_weigth > capacity:
        max_value = 0
    return max_value


def tabu_search(weights, values, capacity, iterations=20):
    tabu_list_len = 5

    current_solution = np.random.choice(
        np.arange(1, 6), size=len(weights), replace=False)
    best_solution = np.random.choice(
        np.arange(1, 6), size=len(weights), replace=False)
    best_value = 0

    tabu_list = np.int64([])

    for i in range(iterations):
        neighbors = np.int64([])
        for j in range(len(weights)):
            neighbor = np.copy(current_solution)
            # print(j)
            # print(neighbor)
            # print(current_solution)
            if neighbor[j] == 0:
                neighbor[j] = 1
            else:
                neighbor[j] = 0
            neighbors = np.append(neighbors, neighbor)
        neighbors = neighbors.reshape(5, 5)
        best_neighbor = None
        best_neighbor_value = -1
        for neighbor in neighbors:
            # print(neighbors)
            neighbor_value = fitness(neighbor, weights, values, capacity)
            print(neighbor)
            print(tabu_list)
            print(
                f"neighbor_value: {neighbor_value} | best_neighbor_value: {best_neighbor_value}")
            print(
                f"first condition: {neighbor_value > best_neighbor_value}\nsecond condition: {not np.any(np.all(neighbor == tabu_list, axis=1))}")
            if neighbor_value > best_neighbor_value and not np.any(np.all(neighbor == tabu_list, axis=1)):
                print("entrou")
                best_neighbor = np.copy(neighbor)
                best_neighbor_value = neighbor_value
                # print(best_neighbor)

        current_solution = np.copy(best_neighbor)
        current_value = best_neighbor_value
        if current_value > best_value:
            best_solution = np.copy(current_solution)
            best_value = current_value

        if len(tabu_list) == tabu_list_len:
            np.delete(tabu_list, 0)
        tabu_list = np.append(tabu_list, current_solution)
        # print(tabu_list)

    return best_solution, best_value


def main():
    weights = np.array([5, 3, 2, 1, 4])
    values = np.array([10, 7, 4, 3, 8])
    # weights = np.arange(1, 6)
    # np.random.shuffle(weights)
    print(weights)
    # values = np.random.choice(np.arange(1,11), size=5, replace=False)
    print(values)
    # iterations = 20
    capacity = 10

    teste = np.int64([0, 3, 1, 5, 2])
    teste2 = np.int64([])

    teste3 = [0, 3, 1, 5, 2]
    teste4 = [[0, 3, 1, 5, 2]]

    print(teste)
    print(teste2)
    print(teste2.any())

    # print(not np.any(np.all(teste == teste2, axis=1)))
    # print(teste3 not in teste4)
    # ts_solution, ts_value = tabu_search(weights, values, capacity)
    # print(f"Solução encontrada: {ts_solution}")
    # print(f"Valor: {ts_value}")


if __name__ == "__main__":
    main()
