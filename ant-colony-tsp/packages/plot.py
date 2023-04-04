"""
Alunos:
Stany Helberth de Souza Gomes da Silva - RA112683
Gabriel de Souza Vendrame - RA112681
Henrique Ribeiro Favaro - 115408
"""

import operator

import matplotlib.pyplot as plt


def plot(pontos, caminho: list):
    x = []
    y = []

    for ponto in pontos:
        x.append(ponto[0])
        y.append(ponto[1])

    y = list(map(operator.sub, [max(y) for i in range(len(pontos))], y))
    plt.plot(x, y, 'co')

    for _ in range(1, len(caminho)):
        i = caminho[_ - 1]
        j = caminho[_]
        plt.arrow(x[i], y[i], x[j] - x[i], y[j] - y[i], color='r', length_includes_head=True)

    plt.xlim(0, max(x) * 1.1)

    plt.ylim(0, max(y) * 1.1)
    plt.show()
