from heapq import heappush, heappop
import time

class Knapsack():
    def __init__(self, state, v, w, d):
        self.state = state
        self.v = v
        self.w = w
        self.d = d

    def __lt__(self, other):
        return self.v + self.d < other.v + other.d
    
    def __str__(self):
        return f'state: {self.state} v: {self.v} w: {self.w} d: {self.d}'
    
    def __repr__(self):
        return f'state: {self.state} v: {self.v} w: {self.w} d: {self.d}'
        
def a_star(W, weight, values):
    n = len(values)

    q = []

    for i in range(n):
        lst = [0] * n
        lst[i] = 1
        heappush(q, (1 / values[i], Knapsack(lst, values[i], weight[i], 0)))

    best = Knapsack([], 0, 0, 0)
    
    while len(q) > 0:
        at = heappop(q)
        atState = at[1]

        if atState.v > best.v:
            best = atState

        for i in range(n):
            if atState.state[i] == 0 and atState.w + weight[i] <= W:
                lst = atState.state.copy()
                lst[i] = 1
                # q = lista de prioridades
                # g(n) = atState.d + 1
                # h(n) = 1 / (atState.v + values[i])
                # f(n) = g(n) + h(n)
                heappush(q, (atState.d + 1 + (1 / (atState.v + values[i])), Knapsack(lst, atState.v + values[i], atState.w + weight[i], atState.d + 1)))

    return best
        
def main():
    initial_time = time.time()

    w = 269
    weight = [95, 4, 60, 32, 23, 72, 80, 62, 65, 46]
    values = [55, 10, 47, 5, 4, 50, 8, 61, 85, 87]

    ans = a_star(w, weight, values)
    print(f"Capacidade da mochila: {w}\nPeso dos itens: {weight}\nValor dos itens: {values}")
    print()
    print("Solução:")
    print(ans)

    end_time = time.time()

    print(f'Tempo: {end_time - initial_time} segundos')


if __name__ == "__main__":
    main()
