from sklearn.neural_network import MLPClassifier
from time import time


gato1 = [1,0,0,0]
gato2 = [0,1,0,0]
gato3 = [0,0,1,0]
cachorro1 = [1,1,1,0]
cachorro2 = [0,1,1,0]
cachorro3 = [1,1,1,1]
treino_x = [gato1,gato2,gato3,cachorro1,cachorro2,cachorro3]
treino_y = [0,0,0,1,1,1]

inicio = time()
modelo = MLPClassifier(hidden_layer_sizes=(500,500,500,500), max_iter=10000)
modelo.fit(treino_x,treino_y)
fim = time()
print(f"tempo decorrido: {fim - inicio}")

animal_misterio = [0,0,0,1]
animal_previsto = modelo.predict([animal_misterio])

if animal_previsto == 0:
    print("O animal é um Gato")
else:
    print("O animal é um Cachorro")