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

# Treinando o modelo, via API
# A biblioteca XGBoost funciona com base em arvores de decisão
modelo = xgb.XGBRegressor(objective='reg:squarederror', enable_categorical=True, n_estimators=100)

modelo.fit(X_treino, y_treino)
preds = modelo.predict(X_teste)

# Calculando o MSE, que é basicamente o quanto o modelo erra
mse = mean_squared_error(y_teste, preds)
print(f"MSE: {mse:.2f}")
print(f"RMSE: {math.sqrt(mse):.2f}")

# Visualizando como o treinamento esta indo, em um gráfico de dispersão
sns.regplot(x=y_teste, y=preds)
plt.xlabel("Valores reais")
plt.ylabel("Valores previstos")
plt.title("Valores previstos X reais")
plt.show()