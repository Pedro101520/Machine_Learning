import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
import seaborn as sns

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

params = {'objective' : 'reg:squarederror'}
# Padronizando o numero de arvores de regreção a floresta aleatoria do XGBoost vai ter
n = 1000

# Comparando o treino com o teste
# Como o XGBoost funciona na base de floresta aeatoria, mas com a diferença que, ele vai melhorando a performance de acordo com a execução
evals = [(dtreino, 'treino'), (dteste, 'validacao')]
modelo = xgb.train(
    params=params,
    dtrain = dtreino,
    num_boost_round = n,
    evals = evals,
    # O parametro abaixo, serve pra exibir os valores a cada 10
    # verbose_eval = 10,
    # Aqui estou adicionando um parametro que faz com que a criação de arvores de regreção sejam interrompidas, caso o nivel de melhora não seja tão significativo. Essa função interrompe de forma aleatoria, quando o que eu disse for identificado
    early_stopping_rounds = 10
)