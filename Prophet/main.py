import pandas as pd
import plotly.express as px
from prophet import Prophet
import numpy as np
from prophet.plot import plot_plotly
from prophet.plot import plot_components_plotly
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math

df = pd.read_csv('Prophet\\poluentes (1).csv')

# Convertendo a coluna referente a data para o formato Datetime
df['Data'] = pd.to_datetime(df['Data'])

fig = px.line(df, x='Data', y='O3')
# fig.show()

# Ajustando as colunas que o prophet vai usar para realizar as previsões
df_prophet = pd.DataFrame()
df_prophet['ds'] = df['Data']
df_prophet['y'] = df['O3']
 
# Separando as informações em treino e teste
df_treino = pd.DataFrame()
df_treino['ds'] = df_prophet['ds'][:1168]
df_treino['y'] = df_prophet['y'][:1168]

df_teste = pd.DataFrame()
df_teste['ds'] = df_prophet['ds'][1168:]
df_teste['y'] = df_prophet['y'][1168:]

# Treinando o modelo
np.random.seed(4587)
modelo = Prophet()
modelo.fit(df_treino)
# periods é igual ao tamanho do teste
futuro = modelo.make_future_dataframe(periods=292, freq='D')
previsao = modelo.predict(futuro)

fig1 = modelo.plot(previsao)
plt.plot(df_teste['ds'], df_teste['y'], '.r')
# plt.show()

# yhat é o valor que o modelo está prevendo
# yhat_lower é o intervalo de confiança inferior
# yhat_upper é o intervalo de confiança superior
previsao[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
print(previsao)

# Gráfico para analisar as caracteristicas da série temporal
modelo.plot_components(previsao)
# plt.show()

# Juntando as datas com os valores previstos, esses valores já são de conhecimento, porém eles servem para medir o quão bem o modelo esta performando
# Ele está pegando apenas as datas que são em comum entre os dois
df_previsao = previsao[['ds', 'yhat']]
df_comparacao = pd.merge(df_previsao, df_teste, on='ds', how='inner')

print(df_comparacao)

# Verificando por meio de uma metrica chamada MSE, o quão bem o modelo está performando
mse = mean_squared_error(df_comparacao['y'], df_comparacao['yhat'])
rmse = math.sqrt(mse)
print(f"RMSE = {rmse}")