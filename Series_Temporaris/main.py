import pandas as pd
import matplotlib.pyplot as plt
from babel.dates import format_date

dados = pd.read_csv('Series_Temporaris\\clientes_restaurantes.csv')
datas_comemorativas = pd.read_csv('Series_Temporaris\\datas_comemorativas.csv')

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

