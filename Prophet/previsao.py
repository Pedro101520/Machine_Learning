import pandas as pd
import json
from prophet.serialize import model_from_json

# Carregar o modelo salvo
with open('Prophet\\modelo.json', 'r') as file_in:
    modelo_carregado = model_from_json(json.load(file_in))

# Carregar os dados originais para obter a última data usada no treinamento
df = pd.read_csv('Prophet\poluentes (1).csv')
df['Data'] = pd.to_datetime(df['Data'])

df_prophet = pd.DataFrame()
df_prophet['ds'] = df['Data']
df_prophet['y'] = df['O3']

# Removendo outliers para garantir que usamos os dados corretos
previsao_treino = modelo_carregado.predict(df_prophet)
sem_outliers = df_prophet[(df_prophet['y'] > previsao_treino['yhat_lower']) & (df_prophet['y'] < previsao_treino['yhat_upper'])]
sem_outliers.reset_index(drop=True, inplace=True)

# Obtendo a última data do conjunto de treino
ultima_data = sem_outliers['ds'].max()
print(f"Última data do treino: {ultima_data}")

# Criando o dataframe futuro para os próximos 80 dias a partir da última data do treino
futuro = pd.date_range(start=ultima_data, periods=81, freq='D')[1:]
futuro = pd.DataFrame({'ds': futuro})

# Fazendo a previsão
previsao = modelo_carregado.predict(futuro)

# Selecionando apenas as colunas de interesse
previsao_80_dias = previsao[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

# Salvando em CSV
previsao_80_dias.to_csv('previsao_80_dias.csv', index=False)
print("Previsão para os próximos 80 dias salva em 'previsao_80_dias.csv'.")
