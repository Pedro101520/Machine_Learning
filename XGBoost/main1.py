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

# Treinamento do modelo
modelo = xgb.train(
    params=params,
    dtrain=dtreino,
    # Setando o numero de arvores que vão ser criadas. O XGBoost vai adequando os ultimos resultados
    num_boost_round=100
)

preds = modelo.predict(dteste)

mse = mean_squared_error(dteste.get_label(), preds)
rmse = math.sqrt(mse)
print(f'RMSE: {rmse:.2f}')

xgb.plot_importance(modelo)
plt.grid(False)
plt.show()