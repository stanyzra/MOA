"""
Alunos:
Stany Helberth de Souza Gomes da Silva - RA112683
Gabriel de Souza Vendrame - RA112681
Henrique Ribeiro Favaro - 115408
"""

from anneal import SimAnneal
import matplotlib.pyplot as plt
import random


def read_coords(path):
    coords = []
    with open(path, "r") as f:
        for line in f.readlines():
            line = [float(x.replace("\n", "")) for x in line.split(" ")]
            coords.append(line)
    return coords


def generate_random_coords(num_nodes):
    return [[random.uniform(-1000, 1000), random.uniform(-1000, 1000)] for i in range(num_nodes)]


if __name__ == "__main__":
    coords = read_coords("coord.txt")  # generate_random_coords(100)
    # coords = generate_random_coords(300)
    sa = SimAnneal(coords, stopping_iter=150000, stopping_T=1e-60, alpha=0.999)
    sa.anneal()
    sa.visualize_routes()
    sa.plot_learning()
    print("Iteração de Parada: ", sa.iteration)
    print("Temperatura de Parada: ", sa.T)
    print("Melhor Solução: ", sa.best_solution)
