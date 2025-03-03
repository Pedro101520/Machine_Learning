import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.ar_model import AutoRegResults
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.ar_model import ar_select_order
from statsmodels.tsa.arima.model import ARIMA
import itertools
from statsmodels.tsa.statespace.sarimax import SARIMAX

# PARA ACHAR O VALOR DO Q, É IGUAL O DO P SÓ QUE NO GRÁFICO DE AUTOCORRELATION, ONDE DEVE-SE PEGAR UM PONTO ANTES DE ENTRAR NA AREA AZUL


# Código para uma séris temporal não estacionaria

# Ignorar avisos
import warnings
warnings.filterwarnings('ignore')

dados_f1 = 'https://raw.githubusercontent.com/alura-cursos/s-ries-temporais-statsmodels/main/Dados/Temperatura_mensal_F1.csv'
dados_f2 = 'https://raw.githubusercontent.com/alura-cursos/s-ries-temporais-statsmodels/main/Dados/Temperatura_mensal_F2.csv'
dados_f3 = 'https://raw.githubusercontent.com/alura-cursos/s-ries-temporais-statsmodels/main/Dados/Temperatura_mensal_F3.csv'

# Uma série estacionaria, é quando os valores não tem uma grande evolução nos valores
# Já uma séries não estacionaria, é quando os valores apresdentam uma evolução, que pode ser pra cima ou pra baixo 
def estac(df):
    # adf verifica se teve ou não tendencia
    adf = adfuller(df)
    print(f"Valor-p do Teste ADF: {adf[1]:.4f}")
    if adf[1] > 0.05:
        print("Não rejeitar a Hipótese Nula: A série não é estacionária\n")
    else:
        print("Rejeitar a Hipótese Nula: A série é estacionária\n")
    
    # KPSS verifica se a série é estacionaria
    kpss_saida = kpss(df)
    print(f"Valor-p do Teste KPSS: {kpss_saida[1]:.4f}")
    if kpss_saida[1] > 0.05:
        print("Não rejeitar a Hipótese Nula: A série é estacionária\n")
    else:
        print("Rejeitar a Hipótese Nula: A série não é estacionária\n")

def plot_prev(treino, teste, mod, nome_mod = ''):
    previsoes = mod.predict(len(treino), len(treino) + len(teste) - 1, dynamic=False)
    plt.figure(figsize=(12,5))
    plt.plot(teste.index, teste, label="Esperado")
    plt.plot(previsoes.index, previsoes, label="Previsto", color="red")

    plt.title(f'Previsão modelo {nome_mod}')
    plt.ylabel('Temperatura {nome_mod}')
    plt.legend()
    plt.show()

    print('\nMetricas:\n')
    mae = mean_absolute_error(teste, previsoes)
    print(f'MAE: {mae}')

    mse = mean_squared_error(teste, previsoes)
    print(f'MSE: {mse}')

df_f3 = pd.read_csv(dados_f3)

df_f3['DATA'] = pd.to_datetime(df_f3['DATA'], format='%Y-%m-%d')
df_f3.set_index(['DATA'], inplace=True, drop=True)
fig = df_f3.plot(figsize=(12, 6))
# plt.show()

# Verificando se é estacionaria ou não
# estac(df_f3)

# Usando o modelo ARIMA, para corrigir e treianr da melhor forma essa série temporal não estacionaria
divisao = int(len(df_f3)*0.8)

treino = df_f3.iloc[:divisao].asfreq('MS')
teste = df_f3.iloc[divisao:].asfreq('MS')

# Gradico para definir os valores pro SARIMA
fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(211)
# Foi colado 48, para ter uma visão maior da sazonalidade
fig = plot_acf(treino, lags=48, ax=ax1)
ax2 = fig.add_subplot(212)
fig = plot_pacf(treino, lags=48, ax=ax2)
plt.show()

# Criando o modelo que consegue lidar com a sazonalidade
# O P é 11, pra ser inferiror ao da sazonalidade
sarima_mod = SARIMAX(treino, order = (11,1,2), seasonal_order=(2,1,1,12)).fit()
print(sarima_mod.summary())

plot_prev(treino, teste, sarima_mod, 'SARIMA(11,1,2)(2,1,1,12)')
plt.show()

mod_f3 = AutoReg(df_f3, 34, old_names=False).fit()
previsao_f3 = mod_f3.predict(len(df_f3), len(df_f3) + 35, dynamic=False)

print(previsao_f3)