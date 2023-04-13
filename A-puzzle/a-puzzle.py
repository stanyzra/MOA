import time

"""
Alunos:
Stany Helberth de Souza Gomes da Silva - RA112683
Gabriel de Souza Vendrame - RA112681
Henrique Ribeiro Favaro - 115408
"""

class Node:
    def __init__(self,data,level,fval):
        """ 
            Inicializa o nó com os dados, o nível do nó e o fvalue calculado 
        """
        self.data = data
        self.level = level
        self.fval = fval

    def generate_child(self):
        """ 
            Gera nós filhos a partir do nó dado movendo o espaço em branco
            em quatro direções {up,down,left,right} 
        """                
        x,y = self.find(self.data,'_') # Encontra a posição do espaço em branco
        """
        val_list contem os valores de posição para mover o espaço em branco em qualquer uma
            das 4 direções [up,down,left,right] respectivamente. 
        """
        val_list = [[x,y-1],[x,y+1],[x-1,y],[x+1,y]]
        children = []
        for i in val_list:
            child = self.shuffle(self.data,x,y,i[0],i[1])
            if child is not None:
                child_node = Node(child,self.level+1,0)
                children.append(child_node)
        return children
        
    def shuffle(self,puz,x1,y1,x2,y2):
        """ 
            Move o espaço em branco na direção dada e se os valores de posição estiverem 
            fora dos limites retorna None
        """
        if x2 >= 0 and x2 < len(self.data) and y2 >= 0 and y2 < len(self.data):
            temp_puz = []
            temp_puz = self.copy(puz)
            temp = temp_puz[x2][y2]
            temp_puz[x2][y2] = temp_puz[x1][y1]
            temp_puz[x1][y1] = temp
            return temp_puz
        else:
            return None
            

    def copy(self,root):
        """ 
            Função de copia para criar uma matriz similar ao nó dado 
        """
        temp = []
        for i in root:
            t = []
            for j in i:
                t.append(j)
            temp.append(t)
        return temp    
            
    def find(self,puz,x):
        """
            Função usada para encontrar a posição do espaço em branco
        """
        for i in range(0,len(self.data)):
            for j in range(0,len(self.data)):
                if puz[i][j] == x:
                    return i,j


class Puzzle:
    def __init__(self,size=3):
        """
            Inicializa o tamanho do quebra-cabeça pelo tamanho especificado
        """
        self.n = size
        self.open = []
        self.closed = []

    def accept(self, auto_input=None):
        """
            Recebe o quebra-cabeça do usuário
        """
        puz = []
        for i in range(0,self.n):
            temp = auto_input[i] if auto_input else input().split(" ")
            puz.append(temp)
        return puz

    def f(self,start,goal):
        """
            Função heurística para calcular o valor heurístico f (x) = h (x) + g (x)
        """
        return self.h(start.data,goal)+start.level

    def h(self,start,goal):        
        """ 
            Calcula a diferença entre os quebra-cabeças dados
        """
        temp = 0
        for i in range(0,self.n):
            for j in range(0,self.n):
                if start[i][j] != goal[i][j] and start[i][j] != '_':
                    temp += 1
        return temp
        

    def process(self, auto_input1=None, auto_input2=None):
        """
            Coleta o estado inicial para a matriz do quebra-cabeça
        """
        if auto_input1 and auto_input2:
            start = self.accept(auto_input1)
            goal = self.accept(auto_input2)
        else:
            print("Exemplo de matriz de dados utilizada: \n1 2 3 \n4 _ 6 \n7 8 5 \n")
            print("Insira a matriz inicial, linha a linha, com os números separados por um espaço em branco, e o espaço vazio representado por _ , como no exemplo dado \n")
            start = self.accept()
            print("Insira a matriz para o objetivo final do quebra-cabeça \n")        
            goal = self.accept()

        initial_time = time.time()
        start = Node(start,0,0)
        start.fval = self.f(start,goal)
        """
            Coloca o nó inicial na lista aberta
        """
        self.open.append(start)
        print("\n\n")
        while True:
            cur = self.open[0]
            print("")
            print("  | ")
            print("  | ")
            print(" \\\'/ \n")
            for i in cur.data:
                for j in i:
                    print(j,end=" ")
                print("")
            """ 
                Se a diferença entre o nó atual e o nó objetivo for 0, chegamos ao nó objetivo 
            """
            if(self.h(cur.data,goal) == 0):
                return initial_time
            for i in cur.generate_child():
                i.fval = self.f(i,goal)
                self.open.append(i)
            self.closed.append(cur)
            del self.open[0]

            """
                Ordena a lista aberta com base no valor f
            """
            self.open.sort(key = lambda x:x.fval,reverse=False)
            
""" 
    Se desejar testar de modo automático, apenas deixe automatic_mode = True
    Caso deseje testar manualmente, deixe automatic_mode = False e insira os dados manualmente quando solicitado
"""
automatic_mode = True

def main():
    if automatic_mode:
        initial_state = [['1', '2', '3'], ['_', '4', '6'], ['7', '5', '8']]
        solution = [['1', '2', '3'], ['4', '5', '6'], ['7', '8', '_']]
    else:
        initial_state = []
        solution = []
        
    # Inicializa o Puzzle 3x3
    puz = Puzzle()
    
    if len(initial_state) > 0 and len(solution) > 0:
        initial_time = puz.process(auto_input1=initial_state, auto_input2=solution)
    else: 
        initial_time = puz.process()
    end_time = time.time()
    
    print("Tempo de execução: ", end_time - initial_time, "segundos")

if __name__ == "__main__":
    main()