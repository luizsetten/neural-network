import numpy as np
from numpy.core.fromnumeric import shape
import scipy

# Algoritmo de treinamento:

# Entrada: X, Y, m, α  ## (X - dados; Y - rotulos; m - quantidade de elementos; α - taxa de aprendizado)

# Inicializar W[1], W[2], b[1] e b[2] aleatoriamente


def inicializa_com_randoms(dim, nr_neuronios):  # nr_neuronios da camada atual
    """
    Cria uma matriz w de dimensoes [dim][1], ou seja, um vetor em coluna, e o inicializa com zero. Inicializa b = 0.

    Parametros:
    dim -- o tamanho do vetor de coluna w (igual a quantidade de caracteristicas de x^(i))

    Retorno:
    w -- vetor coluna inicializado com dimensoes (dim, 1)
    b -- escalar inicializado com valor 0 (vies)
    """

    # cria um vetor de determinada dimensao e seta suas posicoes valendo 0
    w = np.random.rand(dim, nr_neuronios)
    b = np.random.rand(nr_neuronios, 1)

    # garante que o vetor tenha sido criado com as dimensoes adequadas
    assert(w.shape == (dim, nr_neuronios))
    # garante que b eh um float ou int
    assert(b.shape == (nr_neuronios, 1))

    return w, b


def treina(w, b, X, Y, num_iteracoes, taxa_aprendizado, print_custo=False):
    """
    Otimiza w e b com base nos gradientes aprendidos, atraves do algoritmo de gradiente descendente

    Parametros:
    w -- vetor de pesos de tamanho (n_x,1) 
    b -- vies, um escalar
    X -- conjunto de dados de dimensoes (n_x,m) -- instancias sao vistas como vetores-colunas
    Y -- um vetor contendo o ground truth (0 ou 1) de tamanho (1,m)
    num_iteracoes -- numero de interacoes (epocas) para aprendizado do modelo
    taxa_aprendizado -- taxa de aprendizado a ser utilizada
    print_custo -- se verdade, imprimir custo a cada 10 epocas

    Retorno:
    parametros -- dicionario contendo os pesos w e b
    gradientes -- dicionario contendo os gradientes de w e b
    custos -- lista de todos os custos computados durante a treinamento. Utilize essa lista para plotar a curva de aprendizado
    """

    custos = []

    for i in range(num_iteracoes):

        # Realiza uma passagem de propagacao pra frente e pra tras, obtem gradientes
        gradientes, custo = propaga(w, b, X, Y)
        dw = gradientes["dw"]
        db = gradientes["db"]

        # Atualiza w e b com base no gradiente
        w = w - taxa_aprendizado * dw       # --> operacao vetorial
        b = b - taxa_aprendizado * db

        # Guarda os custos a cada 10 iteracoes (epocas)
        if i % 10 == 0:
            custos.append(custo)

        # Exibe o custo a cada 100 iteracoes, se print_custo == true
        if print_custo and i % 10 == 0:
            print("Custo depois de interacao %i: %f" % (i, custo))

    parametros = {"w": w,
                  "b": b}

    gradientes = {"dw": dw,
                  "db": db}

    return parametros, gradientes, custos


def sigmoide(z):
    """
    Computa a funcao logistica (sigmoide) de z

    Parametros:
    z -- Um escalar ou um vetor numpy de qualquer dimensão

    Retorno:
    s -- sigmoide(z)
    """

    s = 1 / (1 + np.exp(-z))  # np.exp(-z) implementa e^(-z)

    return s


# Enquanto (não convergir) ## Ou por um numero de épocas (no projeto atual vai ser 1000)

def treina2nn(X, Y, m, alfa, num_iteracoes):
    print("teste", X.shape[0])
    w, b = inicializa_com_randoms(X.shape[0], 3)
    W1 = w.T
    b1 = b

    w, b = inicializa_com_randoms(3, 1)
    W2 = w.T
    b2 = b

    # J - função de custo
    # dW[1] e db[1] - gradientes da camada oculta
    # dW[2] e db[2] - gradientes da camada de saída
    for i in range(num_iteracoes):
        J = 0
        dW1 = 0
        dW2 = 0
        # b1 = 0
        # b2 = 0

        # for i in range(m):  # Isso aqui ta iterando mas da pra vetorizar (da um puta ganho de desempenho)
        Y_hat = sigmoide(np.dot(W1, X)+b1)
        termo1 = Y*np.log(Y_hat)
        termo2 = (1-Y)*np.log(1-Y_hat)

        # custo = - 1/m * np.sum(termo1 + termo2)
        # Z1 = (termo1 + termo2)

        Z1 = np.dot(W1, X)+b1
        A1 = sigmoide(Z1)  # Tanh() ou ReLU no lugar da sigmoide
        #     z[1](i) = W[1]x(i) + b[1]; a[1](i) = g(z[1](i)) ## Foward propagation [g(z[1](i)) - pode ser tanto tanh(z) quanto ReLU(z)]

        Z2 = np.dot(W2, A1)+b2
        A2 = sigmoide(Z2)
        #     z[2](i) = W[2]a[1](i) + b[2]; a[2](i) = σ(z[2](i)) ## Foward propagation [σ(z[2](i)) - Função sigmoide da camada de saída, ja foi implementado no outro projeto]

        J1 = -1 * [Y*np.log(A2) + (1 - Y)*np.log(1 - A2)]
        #     J += − [y(i)log a[2](i) + (1 − y(i))log (1 − a[2](i))] ## Calcular o custo

        # Y1 tem que ser o ultiimo indice de Y (debuga isso pra ver se o formato ta certo)
        dZ2 = A2-Y1
        #     dz[2] = a[2](i) − y(1) ## Gradiente da camada de saída (camada externa)

        dW2 += np.dot(dZ2, A1.T)
        #     dW[2] += dz[2]a[1](i)^T ## Agrega todas as instancias (camada externa)

        dB2 += dZ2
        #     db[2] += dz[2] ## Agrega todas as instancias (camada externa)

        # dZ1 = np.dot(W2, dZ2 ...

        #     dz[1] = W[2]^T dz[2] ∗ g[1]′(z[1]) ## Gradiente da camada de saída (camada oculta)
        #     dW[1] += dz[1] x(i)^T ## Agrega todas as instancias (camada oculta)
        #     db[1] += dz[1] ## Agrega todas as instancias (camada oculta)

        #   J /= m; dW[1]/= m; db[1]/= m; dW[2]/= m; db[2]/= m ; ## Tirar as médias

        #   W[2] ≔ W[2] − α dW[2] ## Atualização de custos [W[2] - Vetor original; dW[2] - Matriz gradiente; α - Taxa de aprendizado]
        #   b[2] ≔ b[2] − α db[2] ## Atualização de custos
        #   W[1] ≔ W[1] − α dW[1] ## Atualização de custos
        #   b[1] ≔ b[1] − α db[1] ## Atualização de custos

        # Saída: W[1], W[2], b[1] e b[2] treinados
