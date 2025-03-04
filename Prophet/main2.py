import pandas as pd
import plotly.express as px
from prophet import Prophet
import numpy as np
from prophet.plot import plot_plotly
from prophet.plot import plot_components_plotly
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
import json
from prophet.serialize import model_to_json

df = pd.read_csv('Prophet\\poluentes (1).csv')

# Convertendo a coluna referente a data para o formato Datetime
df['Data'] = pd.to_datetime(df['Data'])

fig = px.line(df, x='Data', y='O3')
# fig.show()

# Ajustando as colunas que o prophet vai usar para realizar as previsões
df_prophet = pd.DataFrame()
df_prophet['ds'] = df['Data']
df_prophet['y'] = df['O3']

# Treinando o modelo sem outliers
np.random.seed(4587)
modelo = Prophet()
modelo.fit(df_prophet)
# periods é igual ao tamanho do teste
futuro = modelo.make_future_dataframe(periods=0, freq='D')
previsao = modelo.predict(futuro)

# Removendo outliers
sem_outliers = df_prophet[(df_prophet['y'] > previsao['yhat_lower']) & (df_prophet['y'] < previsao['yhat_upper'])]
sem_outliers.reset_index(drop=True, inplace=True)

# Separando as informações em treino e teste
tamanho_treino = int(len(sem_outliers) * 0.8)
tamanho_teste = int(len(sem_outliers) * 0.2)

# Agora estou separando em treino e teste, porém sem outliers
# Pelo que entendi outliers são os valores que se distanciam muito do intervalo de confiança, que pode ser analisado pelo gráfico
df_treino_sem_outliers = pd.DataFrame()
df_treino_sem_outliers['ds'] = sem_outliers['ds'][:960]
df_treino_sem_outliers['y'] = sem_outliers['y'][:960]

df_teste_sem_outliers = pd.DataFrame()
df_teste_sem_outliers['ds'] = sem_outliers['ds'][960:]
df_teste_sem_outliers['y'] = sem_outliers['y'][960:]

# Treinando o modelo sem outliers
np.random.seed(4587)
# O parametro changepoint, serve para deixar o modelo mais flexivel em relação a TENDENCIA, fazendo com que ele consiga identificar melhor alguns padrões
# Quanto maior, melhor identifica uma tendencia, quanto menor, mais rigido fica em relação a identificação de tendencias
# O parametro yearly_seasonality serve para fazer com que o modelo entenda melhor a sazonalidade da série temporal, por padrão o valor é 10, mas pode aumentar mais. O recomendavel é 20 se naõ estiver indo bem com 10
modelo = Prophet(changepoint_prior_scale=0.5 ,yearly_seasonality=20)
modelo.fit(df_treino_sem_outliers)
futuro = modelo.make_future_dataframe(periods=365, freq='D')
previsao = modelo.predict(futuro)

# fig = modelo.plot(previsao)
# plt.plot(df_teste_sem_outliers['ds'], df_teste_sem_outliers['y'], '.r')
# plt.show()

# fig1 = modelo.plot(previsao)
# plt.plot(df_teste_sem_outliers['ds'], df_teste_sem_outliers['y'], '.r')
# plt.show()

# Aplicando validação cruzada para ver o quão bem o modelo esta performando
# É 365.25, por conta do ano bisesto
# horizon é o dobro do period
df_cv = cross_validation(modelo, initial='365.25 days', period='45 days', horizon='90 days')
# print(df_cv.head())
df_p = performance_metrics(df_cv)
print(df_p)
# Erro medio
print(df_p['rmse'].mean().round(2))

# Salvando o modelo
with open('modelo.json', 'w') as file_out:
    json.dump(model_to_json(modelo), file_out)



# Métricas para entender o quão bem o modelo está performando
# df_previsao = previsao[['ds', 'yhat']]
# df_comparacao = pd.merge(df_previsao, df_teste_sem_outliers, on='ds', how='inner')

# print(df_comparacao)

# mse = mean_squared_error(df_comparacao['y'], df_comparacao['yhat'])
# rmse = math.sqrt(mse)
# print(f"RMSE = {rmse}")
# exit()


# fig = plot_components_plotly(modelo, previsao)
# fig.show()