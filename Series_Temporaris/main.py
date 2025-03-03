import pandas as pd
import matplotlib.pyplot as plt
from babel.dates import format_date
from scipy.stats import zscore
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

dados = pd.read_csv('Series_Temporaris\\clientes_restaurantes.csv')
datas_comemorativas = pd.read_csv('Series_Temporaris\\datas_comemorativas.csv')

# Retorna os valores que desviam em 3 do desvio-padrão
def detectar_anomalias(coluna):
    dados['zscore'] = zscore(dados[coluna])
    anomalias = dados[(dados['zscore'] > 3) | (dados['zscore'] < -3)]
    return anomalias[[coluna, 'zscore', 'dia_da_semana', 'feriado']]

def plot_decomposicao(decomposicao, title):
    fig, axes = plt.subplots(ncols = 1, nrows = 4, sharex=True, figsize=(20, 8))
    decomposicao.observed.plot(ax = axes[0], title = 'Dados observados')
    decomposicao.trend.plot(ax = axes[1], title = 'Tendência')
    decomposicao.seasonal.plot(ax = axes[2], title = 'Sazonalidade')
    decomposicao.resid.plot(ax = axes[3], title = 'Residuos')
    fig.suptitle(title)
    plt.show()

# Convertendo a coluna data pra datetime
dados['data'] = pd.to_datetime(dados['data'])
dados.set_index('data', inplace=True)

# # Elaborando um gráfico, que mostra a movimentação de clientes 
# # Também vou remover valores nulos
# dados['Chimi & Churri'][dados['Chimi & Churri'].isna()]
# # Plotando o gráfico
# dados['Chimi & Churri'].plot(figsize=(20, 3))
# plt.axvline(x=dados['Chimi & Churri'][dados['Chimi & Churri'].isna()].index[0], color='red')
# # PAssou o index como 1, pois tem dois dados nulos
# plt.axvline(x=dados['Chimi & Churri'][dados['Chimi & Churri'].isna()].index[1], color='red');
# plt.show()

# Acima foi feito um código para verificação e visualização de dados nulos
# Em uma base dados que vão servir para series temporais, não é aconsselhavel somente remover o dado nulo, uam opção é pegar a média de valores que estão entre ele e atribuir ao campo correspondente ao valor nulo
dados = dados.interpolate()
# print(dados.loc["2016-04-04":"2016-04-06"])

# Depois da interpolação, foram gerados valores decimais QUE NESSA SITUAÇÃO EM ESPECIFICO não faz sentido, pois estamos falando do numero de clientes por dia
# Logo deverá ser feita um tratamento nas informações
dados = dados.astype(int)

# Forma de descobrir os valores maximos e minimos de cada restaurante
# print(dados.describe())
# Valor total de clientes que passaram no restaurante, no periodo informado
dados['Total'] = dados.sum(axis=1)


# Encontrando datas comemorativas (Preparar o modelo para sazonalidades)
dados['dia_da_semana'] = [format_date(data, "EEEE", locale="pt_BR") for data in dados.index]

datas_comemorativas['data'] = pd.to_datetime(datas_comemorativas['data'])
datas_comemorativas = datas_comemorativas.set_index('data', drop=True)

# Mesclando duas bases de dados, que no caso, são as bases com as informações e a outra com as datas comemorativas
dados = pd.merge(dados, datas_comemorativas, how='left', left_index=True, right_index=True)


# IMPORTANTE PARA GERAÇÃO DE INSIGHT
# Capturando dados discrepantes e entendeno quando ouve um numero maior de clientes em determinados dias
# anomalias_chimi_churri = detectar_anomalias('Chimi & Churri')
# anomalias_assa_frao = detectar_anomalias('Assa Frão')
# print("Anomalias para o Chimi & Churri")
# print(anomalias_chimi_churri)
# print("\n\n Anomalias para o Assa Frão")
# print(anomalias_assa_frao)


# Visualizando informações sobre a série temporal
# Aqui vai ser decomposto as informações da série temporal em: Tendencia, Sazonalidade e Residuos
# Tendencia, mede o crescimento ou decaimento dos valores em um periodo
# Sazonalidade mede anamalias previsiveis, como por exemplo o valor de clientes cair em finais de semana
# Rediduos são informações da série temporal, que ocorrem dew forma aleatoria
# decomposicao_chimi_churri = seasonal_decompose(dados['Chimi & Churri'])
# tendencia = decomposicao_chimi_churri.trend
# sazonaliadade = decomposicao_chimi_churri.seasonal
# residuos = decomposicao_chimi_churri.resid

# Visualizando a tedencia, sazonaliadade e residuos em forma de gráficos
# plot_decomposicao(decomposicao_chimi_churri, 'Decomposicao Chimi $ Churri')


# Acesando informações que variam muito dos residuos, que no caso são informações aleatorias da serie temporal
# anomalias_resid_chimi_churri = np.where(np.abs(decomposicao_chimi_churri.resid) > 2.5 * np.std(decomposicao_chimi_churri.resid))
# anomalias_resid_chimi_churri = dados.iloc[anomalias_resid_chimi_churri][['Chimi & Churri', 'dia_da_semana', 'feriado']]
# print(anomalias_resid_chimi_churri)


# PRevisão de valores
modelo_chimi_churri = ExponentialSmoothing(dados['Chimi & Churri'], seasonal='additive', seasonal_periods=7, freq='D')
resultado_chimi_churri = modelo_chimi_churri.fit()
previsao_chimi_churri = resultado_chimi_churri.forecast(steps = 14)

# plt.figure(figsize = (20,6))
# plt.plot(dados['Chimi & Churri'].index[-100], dados['Chimi & Churri'].values[-100], label = 'Dados Historicos')
# plt.plot(previsao_chimi_churri.index, previsao_chimi_churri.values, label = 'Previsao')
# plt.title('Previsao Holt Winters para o Chimi & Churri')
# plt.xlabel('Data')
# plt.ylabel('Clientes')
# plt.legend()
# plt.show()

tabela_previsao = pd.DataFrame()
tabela_previsao.index = previsao_chimi_churri.index
tabela_previsao['Previsoa Chimi & Churri'] = previsao_chimi_churri.values

tabela_previsao = tabela_previsao.astype(int)

print(tabela_previsao)