import random

# random.seed(3) # definindo random seed para padronizar resultados em diferentes execuções
class Grafo(object):
    """
    Classe que guarda o grafo à ser percorrido durante as iterações
    """
    def __init__(self, matriz_custo: list, quantidade: int):
        """
        :parametro matriz_custo: matriz que guarda o custo entre as cidades
        :parametro quantidade: quantidade de itens na matriz de custo
        """
        self.matriz = matriz_custo
        self.quantidade = quantidade
        self.feromonio = [[1 / (quantidade * quantidade) for j in range(quantidade)] for i in range(quantidade)]


class AntColony(object):
    """
    Classe responsável por conter uma colônia com formigas para percorrer o grafo e buscar resultados
    """
    def __init__(
        self, 
        qtd_formigas: int, 
        geracoes: int, 
        alpha: float, 
        beta: float, 
        rho: float, 
        q: int, 
        estrategia: int
    ):
        """
        :parametro qtd_formigas:
        :parametro geracoes:
        :parametro alpha: importancia relativa do feromonio
        :parametro beta: importancia relativa da heurística
        :parametro rho: coeficiente residual de feromonio
        :parametro q: intensidade do feromonio
        :parametro estrategia: estrategia de atualizacao de feromonio. 0 - ciclo de formiga, 1 - qualidade de formiga, 2 - densidade de formiga
        """
        self.Q = q
        self.rho = rho
        self.beta = beta
        self.alpha = alpha
        self.qtd_formigas = qtd_formigas
        self.geracoes = geracoes
        self.estrategia_atualizacao = estrategia

    def atualizar_feromonio(self, grafo: Grafo, formigas: list):
        for i, linha in enumerate(grafo.feromonio):
            for j, coluna in enumerate(linha):
                grafo.feromonio[i][j] *= self.rho

                for formiga in formigas:
                    grafo.feromonio[i][j] += formiga.delta_feromonio[i][j]

    def calcular_total(self, grafo: Grafo):
        """
        :parametro grafo:
        """
        grafo_resultados = []
        melhor_custo = float('inf')
        melhor_solucao = []

        for gen in range(self.geracoes):
            formigas = [_Formiga(colonia=self, grafo=grafo) for i in range(self.qtd_formigas)]

            for formiga in formigas:
                for i in range(grafo.quantidade - 1):
                    formiga._selecionar_proximo()

                formiga.custo_total += grafo.matriz[formiga.tabu[-1]][formiga.tabu[0]]

                if formiga.custo_total < melhor_custo:
                    melhor_custo = formiga.custo_total
                    melhor_solucao = [] + formiga.tabu

                # atualizar feromonio
                formiga._atualizar_delta_feromonio()

            self.atualizar_feromonio(grafo=grafo, formigas=formigas)

            grafo_resultados.append((gen,melhor_custo))
            print('geracao #{}, melhor custo: {}'.format(gen, melhor_custo))

        return melhor_solucao, melhor_custo, grafo_resultados


class _Formiga(object):
    """
    Classe que representa cada formiga individual da colonia
    """
    def __init__(
        self, 
        colonia: AntColony, 
        grafo: Grafo
    ):
        self.colonia = colonia
        self.grafo = grafo
        self.custo_total = 0.0
        self.tabu = []  # lista tabu para controle de resultados
        self.delta_feromonio = []  # taxa local de feromonio
        self.permitido = [i for i in range(grafo.quantidade)]  # nós permitidos para próxima selecao
        self.eta = [[0 if i == j else 1 / grafo.matriz[i][j] for j in range(grafo.quantidade)] for i in range(grafo.quantidade)]  # informacoes heuristicas
        start = random.randint(0, grafo.quantidade - 1)  # iniciar de nó aleatório
        self.tabu.append(start)
        self.atual = start
        self.permitido.remove(start)

    def _selecionar_proximo(self) -> None:
        denominador = 0
        for i in self.permitido:
            denominador += self.grafo.feromonio[self.atual][i] ** self.colonia.alpha * self.eta[self.atual][i] ** self.colonia.beta
        
        probabilidades = [0 for i in range(self.grafo.quantidade)]  # probabilidades de mover-se para um nó em proxima iteracao

        for i in range(self.grafo.quantidade):
            try:
                self.permitido.index(i)  # teste se a lista de permitido comtém i
                probabilidades[i] = self.grafo.feromonio[self.atual][i] ** self.colonia.alpha * self.eta[self.atual][i] ** self.colonia.beta / denominador
            except ValueError:
                pass 

        # selecionar próximo nó com roleta de probabilidades
        selecionado = 0
        rand = random.random()

        for i, probabilidade in enumerate(probabilidades):
            rand -= probabilidade
            if rand <= 0:
                selecionado = i
                break

        self.permitido.remove(selecionado)
        self.tabu.append(selecionado)
        self.custo_total += self.grafo.matriz[self.atual][selecionado]
        self.atual = selecionado

    
    def _atualizar_delta_feromonio(self) -> None:
        self.delta_feromonio = [[0 for j in range(self.grafo.quantidade)] for i in range(self.grafo.quantidade)]

        for _ in range(1, len(self.tabu)):
            i = self.tabu[_ - 1]
            j = self.tabu[_]

            if self.colonia.estrategia_atualizacao == 1:  # sistema de qualidade de formiga
                self.delta_feromonio[i][j] = self.colonia.Q

            elif self.colonia.estrategia_atualizacao == 2:  # sistema de densidade de formiga
                self.delta_feromonio[i][j] = self.colonia.Q / self.grafo.matriz[i][j]

            else:  # sistema de ciclo de formiga
                self.delta_feromonio[i][j] = self.colonia.Q / self.custo_total
