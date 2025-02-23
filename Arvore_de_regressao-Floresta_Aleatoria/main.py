import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate, KFold
from sklearn.ensemble import RandomForestRegressor


# url = 'https://raw.githubusercontent.com/Mirlaa/regressao-arvores/main/dados_curso/entregas.csv'
# url_teste = 'https://raw.githubusercontent.com/Mirlaa/regressao-arvores/main/dados_curso/teste_entregas.csv'

dados = pd.read_csv("https://raw.githubusercontent.com/Mirlaa/regressao-arvores/main/dados_curso/entregas.csv")

df = dados.copy()
# Convertendo as datas para datetime
df["data_agendada"] = pd.to_datetime(df["data_agendada"], format="%d/%m/%y")
df["data_entrega"] = pd.to_datetime(df["data_entrega"], format="%d/%m/%y")

# Calculos com datas
df["diferenca_dias_entrega"] = (df["data_entrega"] - df["data_agendada"]).dt.days

df["data_agendada_dias"] = df["data_agendada"].dt.day
df["data_agendada_mes"] = df["data_agendada"].dt.month
df["data_agendada_ano"] = df["data_agendada"].dt.year

df["data_entrega_dias"] = df["data_entrega"].dt.day
df["data_entrega_mes"] = df["data_entrega"].dt.month
df["data_entrega_ano"] = df["data_entrega"].dt.year

# Excluindo data_agendada e data_entrega, pois o modelo de arvore não entende objetos do tipo datetime
df.drop(["data_agendada", "data_entrega"], axis=1, inplace=True)

# Feito, para saber quais colunas tem valores unicos, para depois aplicar o one-hot-encoding da manera certa
# colunas_categoricas = ["id_cliente", "nome_artista", "material", "internacional", "envio_expresso", "instalacao_incluida",
#                        "transporte", "fragil", "pedido_extra_cliente", "localizacao_remota"]
# for column in colunas_categoricas:
#     unique_values = df[column].unique()
#     print(f"Valores únicos na coluna '{column}' \n {len(unique_values)} valores:")
#     print(unique_values)
#     print("=="*45)

# Parte responsavel por aplicar o one-hot-encoding
categoricas = ["material", "internacional", "envio_expresso", "instalacao_incluida", "transporte", "fragil", "pedido_extra_cliente", "localizacao_remota"]
df = pd.get_dummies(df, columns = categoricas,
                    prefix=categoricas,
                    drop_first=True)
df.drop(["id_cliente", "nome_artista"], axis=1, inplace=True)


# Ajustando os dados de treinamento de de teste
# Removendo apagando a coluna que se quer prever os valores, para não interferir no treino do modelo
x = df.drop("custo", axis=1)
y = df["custo"]

X_treino, X_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2, random_state=45)


# Treinando a arvore de regressão
dtr = DecisionTreeRegressor(random_state=45)
# Esse é o modelo
dtr.fit(X_treino, y_treino)

# features_importancias = pd.DataFrame({"Features": dtr.feature_names_in_,
#                                     "Importancia": dtr.feature_importances_}).sort_values(by="Importancia", ascending=True)
# plt.figure(figsize=(10, 6))
# bars = plt.barh(features_importancias["Features"], features_importancias["Importancia"])
# plt.xlabel("Importancia Relativa")
# plt.title("Importancia das caracteristicas")

# plt.show()

# Aqui são os parametros para deixar o modelo de arvore de regressão com uma acertividade boa
# Para ir alterando esses valores, vai-se utilizar o skleanr, pois ele consegue fazer de forma automatica
# Essa parte do param_grid junta do código do sklearn, também é responsável por reduzir o overffiting
param_grid = {
    "max_depth": [6, 8, 10, 15],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [5, 10, 15],
    "max_leaf_nodes": [None, 100, 500]
}

# Aqui é o código sklearn que verifica e escolhes os melhores valores para o param_grid, ai depois de fazer as configurações
# O modelo vai ser treinado novamente. (Vai ser utilizado o parametro para se ativar a validação cruzada, pelo que entendi, o sklearn vai selecionar o melhor modelo)
grid_search_dtr = GridSearchCV(dtr, param_grid, scoring="neg_root_mean_squared_error", cv=5, n_jobs=-1)
grid_search_dtr.fit(X_treino, y_treino)

dtr_otimizado = grid_search_dtr.best_estimator_

# Salvando o melhor modelo (No caso é o modelo que se saiu melhor, após o sklearn escolher os hiper parametros)
# print("Melhores parâmetros:", grid_search_dtr.best_params_) 

# Validação cruzada. O KFold serve para dividir o modelo em K vezes
cv_estrategia = KFold(n_splits=3, shuffle=True, random_state=45)
resultados = cross_validate(dtr_otimizado,x,y,scoring="neg_mean_squared_error", cv=cv_estrategia, return_train_score=True)

# Código para exibir os resultados, após aplicar a validação cruzada
treino_rmse = np.sqrt(-resultados["train_score"])
teste_rmse = np.sqrt(-resultados["test_score"])

# print("Treino RMSE em cada fold:", treino_rmse)
# print("Teste RMSE em cada fold:", teste_rmse)
# print("\nMedia do RMSE no treino:", treino_rmse.mean())
# print("Média do RMSE no teste:", teste_rmse.mean())

# # Exibir as metricas do modelo
# print("Métricas conjunto de treino:")
# print("R2:", r2_score(y_treino, dtr_otimizado.predict(X_treino)))
# print("MAE:", mean_absolute_error(y_treino, dtr_otimizado.predict(X_treino)))
# print("RMSE: ", np.sqrt(mean_squared_error(y_treino, dtr_otimizado.predict(X_treino))))
# print("\n\nMétricas conjunto de teste:")
# print("R2:", r2_score(y_teste, dtr_otimizado.predict(X_teste)))
# print("MAE:", mean_absolute_error(y_teste, dtr_otimizado.predict(X_teste)))
# print("RMSE: ", np.sqrt(mean_squared_error(y_teste, dtr_otimizado.predict(X_teste))))

# # Informações sobre a arvore
# print(F"Números de nós: {dtr_otimizado.tree_.node_count}")
# print(f"Número de folhas: {dtr_otimizado.tree_.n_leaves}")
# print(f"Profundidade máxima: {dtr_otimizado.tree_.max_depth}")

# Floresta aleatória, vão ser treinadas varias arvores de regressão, de acordo com as informações e com as caracteristicas da base
# Para tentar melhorar o modelo, pode-se ir alterando o n_estimators para ver qual que fica melhor
rfr = RandomForestRegressor (n_estimators=100, random_state=45, oob_score=True)
rfr.fit(X_treino, y_treino)

# Aplicando validação cruzada na arvores aleatorias
# Para tentar melhorar o modelo, alterar o n_splits, que é a quantidade de divisões para a validação cruzada
cv_estrategia = KFold(n_splits=3, shuffle=True, random_state=45)
resultados = cross_validate(rfr,x,y,scoring="neg_mean_squared_error", cv=cv_estrategia, return_train_score=True)

# Otimizando as arvores de regressão da florestra aleatoria, modificando os parametros das arvores da floresta
param_grid = {
    "max_depth": [None, 15, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [2, 5],
    "max_leaf_nodes": [400, 550]
}

rf = RandomForestRegressor(random_state=45)

grid_search_rfr = GridSearchCV(rf, param_grid, scoring="neg_root_mean_squared_error", cv=5, n_jobs=-1)
grid_search_rfr.fit(X_treino, y_treino)

rfr_otimizado = grid_search_rfr.best_estimator_

# print("Melhores parametros:", grid_search_rfr.best_params_)

# treino_rmse = np.sqrt(-resultados["train_score"])
# teste_rmse = np.sqrt(-resultados["test_score"])

# print("Média do RMSE no treino:", treino_rmse.mean())
# print("Média do RMSE no teste:", teste_rmse.mean())

# Exibir as metricas do modelo
print("Métricas conjunto de treino:")
print("R2:", r2_score(y_treino, rfr_otimizado.predict(X_treino)))
print("MAE:", mean_absolute_error(y_treino, rfr_otimizado.predict(X_treino)))
print("RMSE: ", np.sqrt(mean_squared_error(y_treino, rfr_otimizado.predict(X_treino))))
print("\n\nMétricas conjunto de teste:")
print("R2:", r2_score(y_teste, rfr_otimizado.predict(X_teste)))
print("MAE:", mean_absolute_error(y_teste, rfr_otimizado.predict(X_teste)))
print("RMSE: ", np.sqrt(mean_squared_error(y_teste, rfr_otimizado.predict(X_teste))))