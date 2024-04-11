from Modelo import treinado, modelo
from sklearn.metrics import accuracy_score, confusion_matrix
import random

# exemplo modelo: gato1 = [1,0,0,0]

m1 = [1,0,0,0]
m2 = [0,0,0,1]
m3 = [1,0,1,1]

teste_x = [m1,m2,m3]
teste_y = [0,0,1]
print(teste_y)
previsoes_y = modelo.predict(teste_x)
acuracia = accuracy_score(teste_y, previsoes_y)

print(f"precisão: {acuracia}")

animal_previsto = treinado.predict([m3])
