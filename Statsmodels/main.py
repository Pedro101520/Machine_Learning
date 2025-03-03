import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.ar_model import AutoRegResults
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.ar_model import ar_select_order
from statsmodels.tsa.arima.model import ARIMA

# Código para uma série temporal com valores estacionarios

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

df_f1 = pd.read_csv(dados_f1)

df_f1['DATA'] = pd.to_datetime(df_f1['DATA'], format='%Y-%m-%d')
df_f1.set_index(['DATA'], inplace=True, drop=True)

# df_f1.plot(figsize=(12, 6))
# plt.show()

# estac(df_f1)

# Divisão dos valores para treinamento
divisao = int(len(df_f1) * 0.70)
treino = df_f1.iloc[:divisao].asfreq('MS')
teste = df_f1.iloc[divisao:].asfreq('MS')

# Modelo Autoregressivo (AR) - Ajustando as informações para o desenvolvimento do modelo
# fig = plt.figure(figsize=(10,8))
# ax1 = fig.add_subplot(211)
# fig = plot_acf(treino, lags=20, ax=ax1)

# É recomendavel, escolher o valor do P, onde o indicador do gráfico abaixo, seja um antes da entrada na parte de valores insignificantes do periodo que é indicado por uma barra azul
# ax2 = fig.add_subplot(212)
# fig = plot_pacf(treino, lags=20, ax=ax2)
# plt.show()


# Como disse acima, o lags que esta um antes de entrar no perido é o lags = 14
# Lembrando que esse primeiro lag, tem que achar analisando o gráfico acima, onde ele vai ser igual ao o ultimo ponto antes de entrar na area azul
ar_mod = AutoReg(treino, 14, old_names=False)
ar_res = ar_mod.fit()
# print(ar_res.summary())


mod_result = AutoRegResults(ar_mod, ar_res.params, ar_res.cov_params())
# fig = mod_result.plot_predict(len(treino), len(treino) + len(teste) - 1)
# plt.show()

# 14, pois é o valor do P
# plot_prev(treino, teste, ar_res, 'AR(14)')

# O código abaixo, calcula o melhor numero de lags, pra esse modelo de regressão
# Aqui foi escolhido para o código buscar o melhor valor, com o máximo de 35
ar_selecao = ar_select_order(treino, 35, old_names=False, ic='aic')
# O escolhido foi o 34
print(ar_selecao.ar_lags)

ar_sel_res = ar_selecao.model.fit()
# FIcar de olho no valor de AIC, pois pode-se basear nele, para ver se o modelo esta melhorando
# print(ar_sel_res.summary())

# Verificando se o modelo melhorou
fig = ar_sel_res.plot_predict(len(treino), len(treino) + len(teste) - 1)
# plt.show()


plot_prev(treino, teste, ar_sel_res, 'AR(34)')

mod_f1 = AutoReg(df_f1, 34, old_names=False).fit()
previsao_f1 = mod_f1.predict(len(df_f1), len(df_f1) + 35, dynamic=False)

print(previsao_f1)
