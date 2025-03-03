import pandas as pd
import plotly.express as px
from prophet import Prophet
import numpy as np
from prophet.plot import plot_plotly
from prophet.plot import plot_components_plotly

df = pd.read_csv('Prophet\\poluentes (1).csv')

# Convertendo a coluna referente a data para o formato Datetime
df['Data'] = pd.to_datetime(df['Data'])

fig = px.line(df, x='Data', y='O3')
# fig.show()

# Ajustando as colunas que o prophet vai usar para realizar as previsões
df_prophet = pd.DataFrame()
df_prophet['ds'] = df['Data']
df_prophet['y'] = df['O3']

# Treinando o modelo
np.random.seed(4587)
modelo = Prophet()
modelo.fit(df_prophet)

futuro = modelo.make_future_dataframe(periods=365, freq='D')
previsao = modelo.predict(futuro)

# Ebidindo o gráfico referente ao desempenho do modelo para prever valores futuros
fig = plot_plotly(modelo, previsao)
# fig.show()

# yhat é o valor que o modelo está prevendo
# yhat_lower é o intervalo de confiança inferior
# yhat_upper é o intervalo de confiança superior
previsao[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
print(previsao)

# Gráfico para analisar as caracteristicas da série temporal
plot_components_plotly(modelo, previsao)
fig.show()