import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
import seaborn as sns
# Essa biblioteca é usada pra fazer os testes dos melhores hiperparametros
from sklearn.model_selection import GridSearchCV
import numpy as np
import main
import joblib

# Versão feita com a API
df = pd.read_csv("XGBoost\\dados_automoveis.csv")

# O XGBoost consegue lidar com dados nulos
# É preciso remover valores duplicados, segue exemplo abaixo
# df.duplicated().sum() # Código para listar a quantidade de valores duplicados
dados_limpos = df.drop_duplicates()
dados_limpos.reset_index(drop=True, inplace=True)

# Convertendo tipo objeto pra tipo categoria, pro XGBoost conseguir entender melhor
df_final = dados_limpos.copy()
object_columns =  df_final.select_dtypes(include='object').columns
df_final[object_columns] = df_final[object_columns].astype('category')

# X vão ser as informações de treinamento, deve-se apagar a coluna que se deseja prever os valores
X = df_final.drop(['Valor($)'], axis=1)
y = df_final['Valor($)']


X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.25, random_state=4256)

# Treinando o modelo usando a API nativa
dtreino = xgb.DMatrix(X_treino, y_treino, enable_categorical=True)
dteste = xgb.DMatrix(X_teste, y_teste, enable_categorical=True)

# Código com validação cruzada

params = {'objective' : 'reg:squarederror'}




# Hiperparametros
# Hiperparâmetros, servem para reduzir o RMSE
# Se quiser melhorar mais ainda, é só ir trocando os parametros, ate que o modelo fique o melhor possivel
# param_grid = {
#     'colsample_bytree': [0.6, 1],
#     'subsample': [0.5, 0.8, 1],
#     'max_depth': [3, 5, 10]
# }

# gbm = xgb.XGBRegressor(enable_categorical=True)

# grid = GridSearchCV(
#     estimator = gbm,
#     param_grid = param_grid,
#     scoring = 'neg_mean_squared_error',
#     cv = 5,
#     verbose = 1
# )

# grid.fit(X, y)

# print("Melhores hiperparâmetros encontrados: ", grid.best_params_)
# print("Menor RMSE encontrado: ", np.sqrt(np.abs(grid.best_score_)))
# -----------------------------------------------------




# Depois de descobrir so melhores hiperparâmetros, passar eles dentro da variavel params
# O padrão do learning_rate é 0.3, pode ir alterando essa taxa, pra ver como que o modelo, fica melhor
# Parea achar os melhores hiperparâmetros, é só executar o código comentado acima
params = {'objective': 'reg:squarederror', 'colsample_bytree': 0.6, 'max_depth': 3, 'subsample': 1,
          'learning_rate': 0.3}

# Aplicando validação cruzada, após a escolha dos melhores hiperparâmetros
cv_resultados = xgb.cv(dtrain=dtreino, params=params, nfold=5, num_boost_round=1000,
                       early_stopping_rounds=10, metrics='rmse', as_pandas=True, seed=4789)

modelo_final = xgb.train(
    params=params,
    dtrain = dtreino,
    # Decobrir executando o código final que está comentado
    num_boost_round = 170,
)

preds = modelo_final.predict(dteste)
mse = mean_squared_error(dteste.get_label(), preds)
rmse = math.sqrt(mse)
print(f'RMSE: {rmse:.2f}')


# Visualização do gráfico de disperção pra ver o quão bem o gráfico está performando
# sns.regplot(x=dteste.get_label(), y=preds)
# plt.xlabel('Valores reais')
# plt.ylabel('Valores previstos')
# plt.title('Valores previstos X reais')
# plt.show()

# print(cv_resultados)
# exit()
# Padronizando o numero de arvores de regreção a floresta aleatoria do XGBoost vai ter
# n = 1000


# Esse gráfico é importante, pois com ele é possivel saber, o quão bem o modelo esta performando e o quçao bem ele está aprendendo
# treino_rmse = evals_result['treino']['rmse']
# validacao_rmse = evals_result['validacao']['rmse']

# plt.plot(treino_rmse, label='Treino')
# plt.plot(validacao_rmse, label='Validacao')
# plt.xlabel('Número de iterações')
# plt.ylabel('RMSE')
# plt.title('RMSE de treino e validação ao longo das iterações')
# plt.legend()
# plt.show()


# Qualquer dúvida, olhar o código main3.py, la´esta completo




# Salvando o modelo, pra prever os dados futuros - Acredito que dá pra fazer com qualquer tipode de modelo com machine leaning
joblib.dump(modelo_final, 'modelo_xgb.pkl')
