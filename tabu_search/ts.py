"""
Alunos:
Stany Helberth de Souza Gomes da Silva - RA112683
Gabriel Vendrame de Souza - RA
Henrique 
"""

import numpy as np


def fitness(solution, weights, values, capacity):
    """
    Realiza a avaliação de uma solução e retorna seu valor máximo

    Parâmetros
    ----------
    solution : ndarray
        Possível solução do problema
    weights : ndarray
        Pesos de cada item
    values : ndarray
        Valores de cada item
    capacity : int
        Capacidade total da mochila
    """

    # Definições iniciais de valor e peso máximo
    max_value = 0
    max_weigth = 0

    # Percorre a solução e avalia cada escolha, ou seja, atualiza
    # os valores de valor e peso máximo que a solução possui
    for i, solution_value in np.ndenumerate(solution):
        if solution_value == 1:
            max_value += values[i]
            max_weigth += weights[i]

    # Verifica se a solução não ultrapassa a capacidade da mochila
    if max_weigth > capacity:
        max_value = 0
    return max_value


def tabu_search(weights, values, capacity, iterations=20, tabu_list_len=5):
    """
    Realiza a busca tabu e retorna a solução para o problema. Note que
    os valores da iteração e do tamanho da lista tabu impacta diretamente
    na solução final obtida pela busca.

    Parâmetros
    ----------
    weights : ndarray
        Pesos de cada item
    values : ndarray
        Valores de cada item
    capacity : int
        Capacidade total da mochila
    iterations : int, optional
        Iterações da busca tabu. Default = 20
    tabu_list_len : int, optional
        Tamanho da lista tabu. Default = 5
    """

    # Definições iniciais
    current_solution = np.random.choice(2, size=len(weights))
    best_solution = np.copy(current_solution)
    best_value = 0

    tabu_list = np.int64([])

    solution_shape = len(weights)

    # Inicialização da busca tabu
    for i in range(iterations):
        # Gerando novas soluções vizinhas
        neighbors = np.int64([])
        # Remove ou adiciona um item na mochila
        for j in range(len(weights)):
            neighbor = np.copy(current_solution)
            if neighbor[j] == 0:
                neighbor[j] = 1
            else:
                neighbor[j] = 0

            # Novo conjunto de soluções possíveis
            neighbors = np.append(neighbors, neighbor)

        # Manipulação de dimensões de narray
        neighbors = neighbors.reshape(solution_shape, solution_shape)

        # Definições iniciais para o melhor vizinho e seu valor
        best_neighbor = None
        best_neighbor_value = -1

        # Percorre as soluções vizinhas encontradas anteriormente e busca pela melhor solução
        for neighbor in neighbors:
            # Realiza a avaliação da solução
            neighbor_value = fitness(neighbor, weights, values, capacity)
            # Verifica se o valor total da solução é melhor que o valor total da melhor solução encontrada até agora,
            # além de também se a solução não está na lista tabu, evitando o ótimo local
            if neighbor_value > best_neighbor_value and (not tabu_list.any() or not np.any(np.all(neighbor == tabu_list, axis=1))):
                best_neighbor = np.copy(neighbor)
                best_neighbor_value = neighbor_value

        # Atualizando a solução atual e a melhor solução encontrada, bem como seu valor total de solução
        current_solution = np.copy(best_neighbor)
        current_value = best_neighbor_value
        if current_value > best_value:
            best_solution = np.copy(current_solution)
            best_value = current_value

        # Atualização da lista tabu
        if len(tabu_list) == tabu_list_len:
            tabu_list = np.delete(tabu_list, 0, axis=0)
        tabu_list = np.append(
            tabu_list, np.expand_dims(current_solution, axis=0) if tabu_list.any() else current_solution, axis=0)
        if tabu_list.ndim != 2:
            tabu_list = np.expand_dims(tabu_list, axis=0)

    return best_solution, best_value


def main():

    # Definição dos valores de cada item
    values = np.array([10, 7, 4, 3, 8])
    # Definição dos pesos de cada item
    weights = np.array([5, 3, 2, 1, 4])

    # Definição da capacidade total da mochila
    capacity = 10

    # Chamada da função que faz a busca tabu
    ts_solution, ts_value = tabu_search(
        weights, values, capacity, iterations=100, tabu_list_len=8)
    print(f"Solução encontrada: {ts_solution}")
    print(f"Valor total: {ts_value}")


if __name__ == "__main__":
    main()
