import collections

def readFile(fileName): 
  file = open(fileName, 'r')
  Lines = file.readlines()
  matrix = []

  for line in Lines:    
    line = line.replace('x', '1')
    line = line.replace('b', '3')
    line = line.replace('o', '0')
    line = line.replace('p0sitive', '1')
    line = line.replace('negative', '0')
    line = line.strip('\n')
    vector = line.split(',')
    # print(line)
    # print(vector)
    matrix.append(vector)

  return matrix

# test_train é o valor percentual do quanto você quer separar para teste (30% teste e 70% treino -> teste_train = 0.3)
def separator(test_train: float, matrix):
  countNegative = 0
  countPositive = 0
  limitNegative = 0
  limitPositive = 0

  X_treino = []
  Y_treino = []

  X_teste = []
  Y_teste = []

  # Faz a contagem
  for vector in matrix:
    if vector[-1] == '0': 
      countNegative +=1
    else:
      countPositive +=1

  for vector in matrix:
    if vector[-1] == '0': 
      limitNegative +=1
      if limitNegative < countNegative*test_train:
        # Verificar se é vector[:-1] para pegar todos os elementos menos o ultimo do vetor
        X_teste.append(vector[:-1])
        Y_teste.append(vector[-1])
      else:
        X_treino.append(vector[:-1])
        Y_treino.append(vector[-1])
    else:
      limitPositive +=1
      if limitPositive < countPositive*test_train:
        # Verificar se é vector[:-1] para pegar todos os elementos menos o ultimo do vetor
        X_teste.append(vector[:-1])
        Y_teste.append(vector[-1])
      else:
        X_treino.append(vector[:-1])
        Y_treino.append(vector[-1])

  return X_treino, Y_treino, X_teste, Y_teste

