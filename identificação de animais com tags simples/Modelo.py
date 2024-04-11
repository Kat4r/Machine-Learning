from sklearn.neural_network import MLPClassifier

gato1 = [1,0,0,0]
gato2 = [0,1,0,0]
gato3 = [0,0,1,0]
cachorro1 = [1,1,1,0]
cachorro2 = [0,1,1,0]
cachorro3 = [1,1,1,1]

treino_x = [gato1,gato2,gato3,cachorro1,cachorro2,cachorro3]
treino_y = [0,0,0,1,1,1]

modelo = MLPClassifier(hidden_layer_sizes=(500,500,500,500), max_iter=10000)
treinado = modelo.fit(treino_x, treino_y)