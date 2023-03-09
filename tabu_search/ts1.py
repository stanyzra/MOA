import random

# Definindo os parâmetros do problema da mochila
pesos = [5, 3, 2, 1, 4]
valores = [10, 7, 4, 3, 8]
capacidade = 10

# Definindo os parâmetros da busca tabu
tam_lista_tabu = 8
num_iteracoes = 20

# Inicializando a solução
solucao_atual = [0] * len(pesos)
melhor_solucao = [0] * len(pesos)
melhor_valor = 0

# Inicializando a lista tabu
lista_tabu = []

# Definindo a função de avaliação


def avaliar(solucao):
    peso_total = 0
    valor_total = 0
    for i in range(len(solucao)):
        if solucao[i] == 1:
            peso_total += pesos[i]
            valor_total += valores[i]
    if peso_total > capacidade:
        valor_total = 0
    return valor_total


# Iniciando as iterações da busca tabu
for i in range(num_iteracoes):
    # Gerando novas soluções vizinhas
    vizinhos = []
    for j in range(len(pesos)):
        vizinho = list(solucao_atual)
        if vizinho[j] == 0:
            vizinho[j] = 1
        else:
            vizinho[j] = 0
        vizinhos.append(vizinho)

    # Avaliando as soluções vizinhas e selecionando a melhor
    melhor_vizinho = None
    melhor_valor_vizinho = -1
    for vizinho in vizinhos:
        valor_vizinho = avaliar(vizinho)
        if valor_vizinho > melhor_valor_vizinho and vizinho not in lista_tabu:
            melhor_vizinho = vizinho
            melhor_valor_vizinho = valor_vizinho

    # Atualizando a solução atual e a melhor solução encontrada
    solucao_atual = melhor_vizinho
    valor_atual = melhor_valor_vizinho
    if valor_atual > melhor_valor:
        melhor_solucao = solucao_atual
        melhor_valor = valor_atual

    # Atualizando a lista tabu
    if len(lista_tabu) == tam_lista_tabu:
        lista_tabu.pop(0)
    lista_tabu.append(solucao_atual)

# Imprimindo a melhor solução encontrada
print("Solução encontrada: ", melhor_solucao)
print("Valor: ", melhor_valor)
