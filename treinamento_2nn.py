import numpy as np
from numpy.core.fromnumeric import shape
import scipy

# Algoritmo de treinamento:

# Entrada: X, Y, m, α  ## (X - dados; Y - rotulos; m - quantidade de elementos; α - taxa de aprendizado)

# Inicializar W[1], W[2], b[1] e b[2] aleatoriamente


def inicializa_com_randoms(dim, nr_neuronios):  # nr_neuronios da camada atual
    # cria um vetor de determinada dimensao e seta suas posicoes valendo 0
    w = np.random.rand(dim, nr_neuronios)
    b = np.random.rand(nr_neuronios, 1)

    # garante que o vetor tenha sido criado com as dimensoes adequadas
    assert(w.shape == (dim, nr_neuronios))
    # garante que b eh um float ou int
    assert(b.shape == (nr_neuronios, 1))

    return w, b


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

# escrever tanh() e ReLU (colocar uma flag para selecionar)


def g(z, tp_ativacao='tanh'):
    if(tp_ativacao == 'ReLU'):
        # Escrever a ReLU aqui
        s = np.greater(z, 0)
        s = s.astype(int)
        return s

    # Tangente Hiperbolica
    s = (np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z))
    return s


def g_linha(z, tp_ativacao='tanh'):
    if(tp_ativacao == 'ReLU'):
        #  Derivada da ReLU
        # Se z = 0 o retorno pode ser 0 ou 1, não importa (por isso não tratei o condicional)
        s = np.greater(0, z)
        s = s.astype(int)
        return s

    # Derivada da tangente hiperbolica
    return 1 - g(z)*g(z)

# Enquanto (não convergir) ## Ou por um numero de épocas (no projeto atual vai ser 1000)


def prediz(W1, b1, W2, b2, X, tp_ativacao):
    '''
    Prediz se uma instancia eh 0 ou 1, utilizando os parametros w e b treinados atraves de regressao logistica

    Parametros:
    w -- vetor de pesos de tamanho (n_x,1) 
    b -- vies, um escalar
    X -- conjunto de dados de dimensoes (n_x,m) -- instancias sao vistas como vetores-colunas

    Retorno:
    Y_predicao -- um vetor numpy contendo as predicoes (0 ou 1) para os exemplos em X, de tamanho (1,m)
    '''
    Z1 = np.dot(W1, X)+b1
    A1 = g(Z1, tp_ativacao)  # Tanh() ou ReLU no lugar da sigmoide
    #     z[1](i) = W[1]x(i) + b[1]; a[1](i) = g(z[1](i)) ## Foward propagation [g(z[1](i)) - pode ser tanto tanh(z) quanto ReLU(z)]

    Z2 = np.dot(W2, A1)+b2
    Y_hat = sigmoide(Z2)

    m = X.shape[1]
    Y_predicao = np.zeros((1, m))

    # forca que w tenha a dimensao de um vetor coluna de X
    # w = w.reshape(X.shape[0], 1)

    # Calcula Y_hat, as probabilidades das instancias de X pertencerem a classe 1
    # --> operacao vetorial, Y_hat tem dimensao (1,m)
    # Y_hat = sigmoide(np.dot(w.T, X)+b)

    for i in range(Y_hat.shape[1]):

        # Converte probabilidades Y_hat para prediçoes
        if (Y_hat[0, i] > 0.5):
            Y_predicao[0, i] = 1
        else:
            Y_predicao[0, i] = 0

    assert(Y_predicao.shape == (1, m))

    return Y_predicao


def treina2nn(X, Y, m, alfa, num_iteracoes, nr_neuronios=3, tp_ativacao='ReLU'):
    w, b = inicializa_com_randoms(X.shape[0], nr_neuronios)
    W1 = w.T
    b1 = b

    w, b = inicializa_com_randoms(nr_neuronios, 1)
    W2 = w.T
    b2 = b

    # J - função de custo
    # dW[1] e db[1] - gradientes da camada oculta
    # dW[2] e db[2] - gradientes da camada de saída
    custos = []
    for i in range(num_iteracoes):
        J = 0
        dW1 = 0
        dW2 = 0
        # b1 = 0
        # b2 = 0

        # for i in range(m):  # Isso aqui ta iterando mas da pra vetorizar (da um puta ganho de desempenho)
        # Y_hat = sigmoide(np.dot(W1, X)+b1)
        # termo1 = Y*np.log(Y_hat)
        # termo2 = (1-Y)*np.log(1-Y_hat)

        # custo = - 1/m * np.sum(termo1 + termo2)

        Z1 = np.dot(W1, X)+b1
        A1 = g(Z1, tp_ativacao)  # Tanh() ou ReLU no lugar da sigmoide
        #     z[1](i) = W[1]x(i) + b[1]; a[1](i) = g(z[1](i)) ## Foward propagation [g(z[1](i)) - pode ser tanto tanh(z) quanto ReLU(z)]

        Z2 = np.dot(W2, A1)+b2
        A2 = sigmoide(Z2)
        #     z[2](i) = W[2]a[1](i) + b[2]; a[2](i) = σ(z[2](i)) ## Foward propagation [σ(z[2](i)) - Função sigmoide da camada de saída, ja foi implementado no outro projeto]
        part1 = Y*np.log(A2)
        part2 = (1 - Y)*np.log(1 - A2)
        J = -1 * (part1 + part2)
        #     J += − [y(i)log a[2](i) + (1 − y(i))log (1 − a[2](i))] ## Calcular o custo

        # Y era pra ser Y1 o valor real da instancia mas acho que da certo assim
        dZ2 = A2-Y
        #     dz[2] = a[2](i) − y(1) ## Gradiente da camada de saída (camada externa)

        dW2 += np.dot(dZ2, A1.T)
        #     dW[2] += dz[2]a[1](i)^T ## Agrega todas as instancias (camada externa)

        dB2 = dZ2
        #     db[2] += dz[2] ## Agrega todas as instancias (camada externa)

        dZ1 = np.dot(W2.T, dZ2) * g_linha(Z1, tp_ativacao)
        #     dz[1] = W[2]^T dz[2] ∗ g[1]′(z[1]) ## Gradiente da camada de saída (camada oculta)

        dW1 += np.dot(dZ1, X.T)
        #     dW[1] += dz[1] x(i)^T ## Agrega todas as instancias (camada oculta)

        dB1 = dZ1
        #     db[1] += dz[1] ## Agrega todas as instancias (camada oculta)

        J = np.sum(J) / m
        dW1 = np.sum(dW1) / m
        dB1 = np.sum(dB1) / m
        dW2 = np.sum(dW2) / m
        dB2 = np.sum(dB2) / m  # Tirar as médias
        #   J /= m; dW[1]/= m; db[1]/= m; dW[2]/= m; db[2]/= m ; ## Tirar as médias

        # Atualização de custos [W2 - Vetor original; dW2 - Matriz gradiente; alfa - Taxa de aprendizado]
        W2 = W2 - alfa * dW2
        b2 = b2 - alfa * dB2  # Atualização de custos
        W1 = W1 - alfa * dW1  # Atualização de custos
        b1 = b1 - alfa * dB1  # Atualização de custos
        #   W[2] ≔ W[2] − α dW[2] ## Atualização de custos [W[2] - Vetor original; dW[2] - Matriz gradiente; α - Taxa de aprendizado]
        #   b[2] ≔ b[2] − α db[2] ## Atualização de custos
        #   W[1] ≔ W[1] − α dW[1] ## Atualização de custos
        #   b[1] ≔ b[1] − α db[1] ## Atualização de custos

    # parametros = {"w": np.dot(W2, W1),
    #               "b": np.dot(b1, b2.T)}
    # custos = J
    # return parametros, custos
        if i % 10 == 0:
            custos.append(np.sum(J)/m)

    return W2, b2, W1, b1, custos
# Saída: W[1], W[2], b[1] e b[2] treinados


def constroi_modelo_2nn(X_treino, Y_treino, X_teste, Y_teste, num_iteracoes, taxa_aprendizado, nr_neuronios, tp_ativacao, print_custo=False):
    """
    Constroi o modelo de regressao logistica com base nas funcoes anteriores

    Parametros:
    X_treino -- conjunto de treinamento de dimensao (n_x, m_treino) que contem as instancias para treinamento
    Y_treino -- rotulos de treinamento de dimensao (1, m_treino)
    X_teste -- conjunto de teste de dimensao (n_x, m_teste) que contem as instancias para teste
    Y_teste -- rotulos de teste de dimensao (1, m_teste)
    num_iteracoes -- numero de interacoes (epocas) para aprendizado do modelo
    taxa_aprendizado -- taxa de aprendizado a ser utilizada
    print_custo -- se verdade, imprimir custo a cada 10 epocas

    Retorno:
    d -- dicionario contendo informacoes sobre o modelo
    """
    m = X_treino.shape[1]
    W2, b2, W1, b1, custos = treina2nn(X_treino, Y_treino, m, taxa_aprendizado,
                                       num_iteracoes, nr_neuronios, tp_ativacao)

    # Prediz o rótulo de treinamento e de teste
    Y_predicao_treino = prediz(W1, b1, W2, b2, X_treino, tp_ativacao)
    Y_predicao_teste = prediz(W1, b1, W2, b2, X_teste, tp_ativacao)

    # Exibe a taxa de acuracia do treino e do teste
    # Taxa de acuracia = 100 - MAE*100
    # Erro medio absoluto (MAE) ~~> MAE = 1/m * soma(|y_predito^(i) - y^(i)|)
    print("Acuracia de treino: {} %".format(
        100 - np.mean(np.abs(Y_predicao_treino - Y_treino)) * 100))
    print("Acuracia de teste: {} %".format(
        100 - np.mean(np.abs(Y_predicao_teste - Y_teste)) * 100))

    d = {"custos": custos,
         #  "Y_predicao_teste": Y_predicao_teste,
         #  "Y_predicao_treino": Y_predicao_treino,
         "W1": W1,
         "b1": b1,
         "W2": W2,
         "b2": b2,
         "taxa_aprendizado": taxa_aprendizado,
         "num_iteracoes": num_iteracoes}
    return d
