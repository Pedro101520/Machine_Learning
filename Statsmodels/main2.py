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

# Código para uma série temporal com residuos

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

# Função responsavel por encontrar os melhores valores para o treinamento do modelo
def grid_arima(p_inicial, p_final, q_inicial, q_final, d_valores, treino):
    # Definindo os parametros
    p_params = range(p_inicial, p_final)
    q_params = range(q_inicial, q_final)
    d_params = [d_valores]

    # Gerando todas as combinações possiveis usando product
    combinacoes = list(itertools.product(p_params, d_params, q_params))
    aic_grid = dict()

    # Treinando o modelo e salvando todas combinações
    for order in combinacoes:
        try:
            model = ARIMA(treino, order=order).fit()
            aic_grid[order] = list()
            aic_grid[order].append((model.aic if model.aic else float('inf')))
        except:
            continue
    return aic_grid, min(aic_grid, key=lambda x: aic_grid[x][0])

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

df_f2 = pd.read_csv(dados_f2)

df_f2['DATA'] = pd.to_datetime(df_f2['DATA'], format='%Y-%m-%d')
df_f2.set_index(['DATA'], inplace=True, drop=True)
fig = df_f2.plot(figsize=(12, 6))
# plt.show()

# Verificando se é estacionaria ou não
estac(df_f2)

divisao = int(len(df_f2)*0.8)

treino = df_f2.iloc[:divisao].asfreq('MS')
teste = df_f2.iloc[divisao:].asfreq('MS')

# É aqui que se identifica o valor de P ,olhando no gráfico de baixo, e eencontradno um ponto antes, do qual começa ficar dentro da area azul
fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(211)
fig = plot_acf(treino, lags=20, ax=ax1)
ax2 = fig.add_subplot(212)
fig = plot_pacf(treino, lags=20, ax=ax2)
# plt.show()

ar_mod_f2 = AutoReg(treino, 10, old_names=False).fit()
print(ar_mod_f2.summary())

# Nesse gráfico, mas especificamente no correlogram, foi possivel identificar que tinha residuos na série temporal, e após identificar isso, olhando se tinham pontos fora da area azul no gráfico que mencionei
# Se identificado residuos, deve-se aplicar o modelo abaixo, baseado no ARIMA
# Se tiver residuo segue o código do main.py
ar_mod_f2.plot_diagnostics(figsize=(15,12))
# plt.show()

# Treinando um modelo, para no caso ele tiver residuos na serie temporal
# Nesse caso o 0 do order é padrão nesse caso
# O 1 é o P, mas vai ser escolhido um melhor no código a seguir
arma_mod = ARIMA(treino, order = (10,0,1)).fit()
print(arma_mod.summary())

aic_arma = grid_arima(10,25,1,3,0,treino)
# Aqui exibe os valores onde o modelo teve o melhor resultado
print(aic_arma[1])

arma_mod_ot = ARIMA(treino, order=aic_arma[1]).fit()
print(arma_mod_ot.summary())

plot_prev(treino, teste, arma_mod_ot, 'ARMA(24,2)')
# plt.show()

arma_mod_ot.plot_diagnostics(figsize=(15,12))
plt.show()

mod_f2 = AutoReg(df_f2, 34, old_names=False).fit()
previsao_f2 = mod_f2.predict(len(df_f2), len(df_f2) + 35, dynamic=False)