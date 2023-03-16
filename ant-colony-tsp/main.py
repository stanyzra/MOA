import math
import random

from packages.antColony import AntColony, Grafo
from packages.plot import plot


# calcula a distancia entre duas cidades para usar como custo
def distancia(cidade1: dict, cidade2: dict):
    return math.sqrt((cidade1['x'] - cidade2['x']) ** 2 + (cidade1['y'] - cidade2['y']) ** 2)

def main():
    # random.seed(3) # definindo random seed para padronizar resultados em diferentes execuções

    cidades = []
    pontos = []

    with open('./data/att48.txt') as f:
        for linha in f.readlines():
            cidade = linha.split(' ')
            cidades.append(
                dict(
                    index=int(cidade[0]), 
                    x=int(cidade[1]), 
                    y=int(cidade[2])
                )
            )
            pontos.append(
                (
                    int(cidade[1]), 
                    int(cidade[2])
                )
            )

    matriz_custo = []
    quantidade = len(cidades)

    for i in range(quantidade):
        linha_tabela = []

        for j in range(quantidade):
            linha_tabela.append(distancia(cidades[i], cidades[j]))

        matriz_custo.append(linha_tabela)
    
    antColony = AntColony(
        qtd_formigas=30, 
        geracoes=500, 
        alpha=1.0, 
        beta=10.0, 
        rho=0.5, 
        q=10, 
        estrategia=2
    )

    grafo = Grafo(
        matriz_custo=matriz_custo, 
        quantidade=quantidade
    )

    caminho, custo, grafo_resultados = antColony.calcular_total(grafo=grafo)

    print('Custo: {}, caminho: {}'.format(custo, caminho))
    plot(pontos=pontos, caminho=caminho) # plotar caminho encontrado

    plot(pontos=grafo_resultados, caminho=[i for i in range(len(grafo_resultados))]) # plotar grafico de resultados durante gerações

if __name__ == '__main__':
    main()
