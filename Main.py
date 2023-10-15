import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

base = pd.read_csv('Base.csv', delimiter=',')

jogos = pd.DataFrame()
jogos['Numeros'] = np.arange(1,61)

verificador =[]
jogos_lista = jogos.iloc[:,0].values
base_numero_ultimo_jogo = int(base.iloc[0,0:1].values)

for i in range(0,2000):
  base_jogo_atual = base.iloc[i,1:].values
  for j in jogos_lista:
    if j in base_jogo_atual:
      verificador.append(1)
    else:
      verificador.append(0)
  numero_jogo = base_numero_ultimo_jogo - i
  jogos[numero_jogo] = verificador
  verificador = []

print(jogos)

colunas_para_transformar = jogos.columns[1:2001]
jogos = pd.melt(jogos, id_vars=jogos.columns[0:1], value_vars=colunas_para_transformar, var_name='Coluna', value_name='Valor')

print(jogos)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

resultado_prob = []

for i in range(1,61):
  jogos_numero = jogos[jogos['Numeros'] == i]
  X = jogos_numero.iloc[ :, 0:1].values
  y = jogos_numero.iloc[ :, -1].values
  model.fit(X, y)
  y_prob = (model.predict_proba([[i]])) 
  resultado_prob.append(y_prob[0][1])

df = pd.DataFrame(resultado_prob,columns=['probabilidade'])
df = df.sort_values(by='probabilidade', ascending=False)
df = df.set_index(df.index + 1)

print(df)





