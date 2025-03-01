import joblib
import pandas as pd
import xgboost as xgb

modelo = joblib.load('XGBoost\\modelo_xgb.pkl')

df = pd.read_csv('https://raw.githubusercontent.com/vqrca/regressao_boosting/main/Dados/novos_automoveis.csv')

# Convertendo os tipos objeto pra categorical
object_columns = df.select_dtypes(include='object').columns
df[object_columns] = df[object_columns].astype('category')

dados_novos = xgb.DMatrix(df, enable_categorical=True)
previsoes = modelo.predict(dados_novos)

df_novos_automoveis = pd.DataFrame(df)
df_novos_automoveis['Previsoes'] = previsoes

print(df_novos_automoveis.head())