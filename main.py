from fileHandler import readFile, separator
from Regressao_Logistica import constroi_modelo
import numpy as np
import matplotlib.pyplot as plt

matrix = readFile('tic-tac-toe.data')

X_treino, Y_treino, X_teste, Y_teste = separator(0.3, matrix)

X_treino = np.array(X_treino).T.astype(np.float)
Y_treino = np.array([Y_treino]).astype(np.float)
X_teste = np.array(X_teste).T.astype(np.float)
Y_teste = np.array([Y_teste]).astype(np.float)

print(X_treino.shape)
print(Y_treino.shape)
print(X_teste.shape)
print(Y_teste.shape)

def raioGraficalizador(dados, intervalo, numeroIteracoes, taxaAprendizado):
  fig,ax = plt.subplots()
  x = [None] * dados.__len__()

  for i in range(dados.__len__()):
    x[i] = i*intervalo

  ax.plot(x, dados)
  plt.scatter(x, dados) # scatter são os quadradinhos na linha

  titulo = 'Taxa de aprendizado: {0} | Número de iterações: {1}'
  titulo = titulo.format(taxaAprendizado, numeroIteracoes)
  plt.title(titulo)
  plt.xlabel('Número de iterações') #definindo nome do eixo X
  plt.ylabel('Custo')
  taxa = str(taxaAprendizado).replace('.','_')
  nomeArquivo = '{0}and{1}.png'
  nomeArquivo = nomeArquivo.format(numeroIteracoes, taxa)
  plt.savefig(nomeArquivo)
  plt.show()

vetor1 = [50, 100, 1000]
vetor2 = [0.1, 0.01, 0.001]

for i in vetor1:
  for j in vetor2:
    data = constroi_modelo(X_treino, Y_treino, X_teste, Y_teste, i, j, print_custo=False)
    raioGraficalizador(data['custos'], 10, i, j)
