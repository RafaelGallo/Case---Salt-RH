#!/usr/bin/env python
# coding: utf-8

# # Case Salt RH

# # Modelo machine learning - Previsão de casas - AMES

# # Exploratory Data Analysis (EDA)
# 

# **--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------**

# **Desafio Técnico**
# 
# - **Questão 1**
# 
# Quando você obtém seus dados pela primeira vez, é muito tentador começar imediatamente a ajustar os modelos e avaliar o desempenho deles. No entanto, antes de começar a modelar, é absolutamente essencial explorar a estrutura dos dados e os relacionamentos entre as variáveis no conjunto de dados.
# Faça um EDA detalhado do conjunto de dados ames_train, para aprender sobre a estrutura dos dados e os relacionamentos entre as variáveis no conjunto de dados.
# 
# 
# - **Questão 2**
# 
# Depois de ter explorado completamente, certifique-se de criar pelo menos quatro gráficos que você achou mais informativos durante seu processo de EDA e explique brevemente o que você aprendeu com cada um (por que você achou cada informativo).
# 
# 
# - **Questão 3**
# 
# Na construção de um modelo, geralmente é útil começar criando um modelo inicial simples e intuitivo com base nos resultados da análise exploratória de dados. Você pode sentir vontade de apresentar habilidades estatísticas mais avançadas. Por esse motivo, estamos fornecendo dados de teste no conjunto de dados ames_test para que você possa construir um modelo simples para prever os preços das casas com base nos dados disponíveis no conjunto de dados de treinamento. Use sua imaginação.
# 

# **Nota: O objetivo não é identificar o “melhor” modelo possível, mas escolher um ponto de partida razoável e compreensível**

# # Regras de Envio
# 
# Para que o seu desafio seja analisado, você deverá atender às seguintes regras:
# 
# - Respeitar a data limite informada no email
# 
# - Fazer um projeto utilizando a linguagem python
# 
# - O código deverá ser entregue como um notebook no formato .ipynb, identificando adequadamente as questões, com comentários e explicações.
# 

# # Avaliação
# 
# Iremos avaliar a forma que você resolveu as questões, sua criatividade, insights e passos que tomou para obtenção de resultados, assim como os comentários e forma como desenvolveu os códigos. Dessa forma sugerimos a originalidade e criatividade para resolução das questões.
# 

# **--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------**

# # 0 - Importação das bibliotecas

# In[1]:


# Versão do python
from platform import python_version

print('Versão python neste Jupyter Notebook:', python_version())


# In[2]:


# Importação das bibliotecas 

import pandas as pd # Pandas carregamento csv
import numpy as np # Numpy para carregamento cálculos em arrays multidimensionais

# Visualização de dados
import seaborn as sns
import matplotlib as m
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly
import plotly.express as px

# Carregar as versões das bibliotecas
import watermark

# Warnings retirar alertas 
import warnings
warnings.filterwarnings("ignore")


# In[3]:


# Versões das bibliotecas

get_ipython().run_line_magic('reload_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-a "Versões das bibliotecas" --iversions')


# In[4]:


# Configuração para os gráficos largura e layout dos graficos

plt.rcParams["figure.figsize"] = (25, 20)

plt.style.use('fivethirtyeight')
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

m.rcParams['axes.labelsize'] = 25
m.rcParams['xtick.labelsize'] = 25
m.rcParams['ytick.labelsize'] = 25
m.rcParams['text.color'] = 'k'


# # 0.1) Base de dados

# In[5]:


# Carregando a base de dados
data_train = pd.read_csv("ames_train.csv", sep = ";")
data_test = pd.read_csv("ames_test.csv", sep = ";")


# # 0.2) Descrição dados
# 
# - Verificação de linhas colunas informaçãos dos dados e tipos de variáveis. Valores das colunas verficando dados nulos ou vazios.

# In[6]:


# Exibido 5 primeiros dados de data_train

data_train.head()


# In[7]:


# Exibido 5 últimos dados de data_train

data_train.tail()


# In[8]:


# Exibido 5 primeiros dados de data_test

data_test.head()


# In[9]:


# Exibido 5 últimos dados de data_train

data_test.tail()


# In[10]:


# Número de linhas e colunas

data_train.shape


# In[11]:


# Verificando informações das variaveis

data_train.info()


# In[12]:


# Exibido tipos de dados

data_train.dtypes


# In[13]:


# Total de colunas e linhas - data_train

print("Números de linhas: {}" .format(data_train.shape[0]))
print("Números de colunas: {}" .format(data_train.shape[1]))


# In[14]:


# Total de colunas e linhas - data_test

print("Números de linhas: {}" .format(data_test.shape[0]))
print("Números de colunas: {}" .format(data_test.shape[1]))


# In[15]:


# Exibindo valores ausentes e valores únicos

print("\nMissing values :  ", data_train.isnull().sum().values.sum())
print("\nUnique values :  \n",data_train.nunique())


# In[16]:


# Exibindo valores ausentes e valores únicos

print("\nMissing values :  ", data_test.isnull().sum().values.sum())
print("\nUnique values :  \n",data_test.nunique())


# # 0.3) - Limpeza da base de dados

# In[17]:


# Dados faltantes coluna óbitos

data = data_train[data_train["price"].notnull()]
data.isna().sum()


# In[18]:


# Removendo dados ausentes do dataset 

data = data_train.dropna()
data_train.head()


# In[19]:


# Sum() Retorna a soma dos valores sobre o eixo solicitado
# Isna() Detecta valores ausentes

data_train.isna().sum()


# In[20]:


# Retorna a soma dos valores sobre o eixo solicitado
# Detecta valores não ausentes para um objeto semelhante a uma matriz.

data_train.notnull().sum()


# In[21]:


# Total de número duplicados

data_train.duplicated()


# In[22]:


# Dados faltantes

data_train.fillna(0, inplace=True)
data_train.head()


# In[23]:


# Períodos faltantes

sorted(data_train['price'].unique())


# # Questão 1
# 
# Quando você obtém seus dados pela primeira vez, é muito tentador começar imediatamente a ajustar os modelos e avaliar o desempenho deles. No entanto, antes de começar a modelar, é absolutamente essencial explorar a estrutura dos dados e os relacionamentos entre as variáveis no conjunto de dados.
# Faça um EDA detalhado do conjunto de dados ames_train, para aprender sobre a estrutura dos dados e os relacionamentos entre as variáveis no conjunto de dados.
# 
# 
# **R**: Primeira etapa que fiz sabe os tipos das variáveis nós dados de ames_train, depois uma limpeza dos dados removendo dados nulos, ausentes e dados duplicados, fiz uma estatística descritiva visualizar como percentil, média, moda, mediana, depois eu fiz uma distribuição normal da coluna preços do imóveis. Uma análise de boxplot verificando possíveis outliers dentro dos dados. 
# 

# # 0.4) Estatística descritiva

# In[24]:


# Exibindo estatísticas descritivas visualizar alguns detalhes estatísticos básicos como percentil, média, padrão, etc. 
# De um quadro de dados ou uma série de valores numéricos.

data_train.describe().T


# # 0.5) Gráfico de distribuição normal

# In[25]:


# Gráfico distribuição normal
plt.figure(figsize=(18.2, 8))

ax = sns.distplot(data_train['price']);
plt.title("Distribuição normal", fontsize=20)
plt.xlabel("Preço do imóvel")
plt.ylabel("Total")
plt.axvline(data_train['price'].mean(), color='b')
plt.axvline(data_train['price'].median(), color='r')
plt.axvline(data_train['price'].mode()[0], color='g');
plt.legend(["Media", "Mediana", "Moda"])
plt.show()


# In[85]:


# Verificando os dados no boxplot valor total verificando possíveis outliers

plt.figure(figsize=(18.2, 8))
ax = sns.boxplot(x="Sale.Condition", y="price", data = data_train)
plt.title("Gráfico de boxplot - Região o valor total")
plt.xlabel("Total")
plt.ylabel("Valor total")


# In[27]:


# Cálculo da média preços dos imóveis 

media_preco = data_train[['price', 'MS.SubClass']].groupby('price').mean()
media_area = data_train[["area", "price"]].groupby('price').mean()

print("Média de Preço", media_preco)
print()
print("Média da Idade", media_area)


# # 0.6) Matriz de correlação dos dados

# In[28]:


# Matriz correlação de pares de colunas, excluindo NA / valores nulos.

corr = data_train.corr()
corr


# In[29]:


# Gráfico da matriz de correlação

plt.figure(figsize=(60.5,45))
ax = sns.heatmap(data_train.corr(), annot=True, cmap='YlGnBu');
plt.title("Matriz de correlação")


# # 0.5) Análise de dados

# # Questão 2
# 
# Depois de ter explorado completamente, certifique-se de criar pelo menos quatro gráficos que você achou mais informativos durante seu processo de EDA e explique brevemente o que você aprendeu com cada um (por que você achou cada informativo).
# 
# **R**: Eu escolhei os quarto gráficos como o preço dos imóveis, data de nascimento das pessoas, região pela condição dos moradores, condição de vinda pelo valor do imóvel. Na minha analise no gráfico 1 e preços do imóvel é possivil sabe o valor da venda do valor baixo e alto . No segundo gráfico séria nascimento das pessoas que nasceram de 1900 até 200 nessa caso podemos observar que nasceram em 1960 comparam imóveis idades. No terceiro gráfico séria as condições das pessoas que são familiar, solteiros, casados. E no quarto gráfico seria a região aonde as pessoas que comparam os imóveis.

# In[30]:


# Observando total dos preços dos imóvel

plt.figure(figsize=(18.2, 8))
sns.histplot(data_train["price"])
plt.title("Preço do imóvel")
plt.xlabel("Valor")
plt.ylabel("Total")


# In[87]:


# Gráfico nascimento das pessoas 
plt.figure(figsize=(18.2, 8))

plt.title("Nascimento das pessoas")
ax = sns.histplot(data_train["Year.Built"])
plt.ylabel("Total")
plt.xlabel("Ano")


# In[32]:


# Gráfico condições de vinda por valor do imóvel
plt.figure(figsize=(25.5, 15))

plt.title("Condições de vinda pelo valor preço dos imóvel")
ax = sns.barplot(x="Yr.Sold", y="price", data = data_train, hue="Sale.Condition")
plt.ylabel("Valor")
plt.xlabel("Imovel vendidos")


# In[33]:


# Região das vendas dos imóveis pela área
plt.figure(figsize=(18.2, 8))

plt.title("Área da cidade")
ax = sns.scatterplot(x="area", y="price", data = data_train, hue = "Sale.Condition")
plt.xlabel("Valor dos imóveis")
plt.ylabel("Valor")


# In[34]:


# Gráfico condição de vida das pessoas 
plt.figure(figsize=(18.2, 8))

plt.title("Condição de venda dos imóveis")
sns.countplot(data_train["Sale.Condition"])
plt.xlabel("Condição")
plt.ylabel("Total")


# # 0.6) Análise de dados = Univariada

# In[83]:


# Fazendo um comparativo dos dados 

data_train.hist(bins = 25, figsize=(40.2, 35))
plt.title("Gráfico de histograma")
plt.show()


# # 0.7) Data Processing
# O processamento de dados começa com os dados em sua forma bruta e os converte em um formato mais legível (gráficos, documentos, etc.), dando-lhes a forma e o contexto necessários para serem interpretados por computadores e utilizados.
# 
# Exemplo: Uma letra, um valor numérico. Quando os dados são vistos dentro de um contexto e transmite algum significado, tornam-se informações.

# In[89]:


# Tipos dos dados
data_test.dtypes


# In[37]:


# Mundando os tipo de dados de object para inteiros 

data_test['Lot.Area'] = data_test['Lot.Area'].astype(int)
data_test['Yr.Sold'] = data_test['Yr.Sold'].astype(int)
data_test.dtypes


# # 0.8) Feature Engineering
# 
# Praticamente todos os algoritmos de Aprendizado de Máquina possuem entradas e saídas. As entradas são formadas por colunas de dados estruturados, onde cada coluna recebe o nome de feature, também conhecido como variáveis independentes ou atributos. Essas features podem ser palavras, pedaços de informação de uma imagem, etc. Os modelos de aprendizado de máquina utilizam esses recursos para classificar as informações.
# Por exemplo, sedentarismo e fator hereditário são variáveis independentes para quando se quer prever se alguém vai ter câncer ou não
# 
# As saídas, por sua vez, são chamadas de variáveis dependentes ou classe, e essa é a variável que estamos tentando prever. O nosso resultado pode ser 0 e 1 correspondendo a 'Não' e 'Sim' respectivamente, que responde a uma pergunta como: "Fulano é bom pagador?" ou a probabilidade de alguém comprar um produto ou não.

# In[38]:


# Importando a biblioteca para pré-processamento 

from sklearn.preprocessing import LabelEncoder

for i in data_test.columns:
    if data_test[i].dtype==np.number:
        continue
    data_test[i]= LabelEncoder().fit_transform(data_test[i])
    
data_test.head(4)


# # 0.9) Treino e Teste
# 
# - Treino e teste da base de dados da coluna price e idade

# In[39]:


x = data_test[["Lot.Area", "Yr.Sold"]] # Variável para treino
y = data_test["price"] # Variável para teste


# In[40]:


# Total de linhas e colunas dados variável x
x.shape


# In[41]:


# Total de linhas e colunas dados variável y
y.shape


# # 10) Escalonamento
# 
# - Escalonamento uma forma de contornar os problemas relacionados à escala, mantendo a informação estatística dos dados. O procedimento consiste em realizar uma transformação sobre o conjunto original dos dados de modo que cada variável apresente média zero e variância unitária.

# In[42]:


# Importando a biblioteca sklearn para o escalonamneto dos dados

from sklearn.preprocessing import StandardScaler 

scaler_pre = StandardScaler() # Inicializando o escalonamento
scaler_pre_fit_train = scaler_pre.fit_transform(x) # Treinamento com a função fit_transform com a variável x
scaler_pre_fit_train # Imprimindo o valor do escalonamento


# # 11) Modelo treinado para x, y valor
# 
# - 20% para os dados de treino
# - 80% para teste
# - Random state igual a zero

# In[43]:


# Importação da biblioteca sklearn para treino e teste do modelo

from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(x, # Variável x
                                                    y, # Variável y
                                                    test_size=0.20, # Divivindo os dados em 20% para treino e 80% para teste
                                                    random_state = 0) # Random state igual a zero


# In[44]:


# Total de linhas e colunas e linhas dos dados de treino x

x_train.shape


# In[45]:


# Total de linhas dos dados de treino y

y_train.shape


# In[46]:


# Total de linhas e colunas dos dados de treino x teste 

x_test.shape


# In[47]:


# Total de linhas e colunas dos dados de treino y teste 

y_test.shape


# # 12) Modelo machine learning 

# # Questão 3
# 
# Na construção de um modelo, geralmente é útil começar criando um modelo inicial simples e intuitivo com base nos resultados da análise exploratória de dados.
# Você pode sentir vontade de apresentar habilidades estatísticas mais avançadas. Por esse motivo, estamos fornecendo dados de teste no conjunto de dados ames_test para que você possa construir um modelo simples para prever os preços das casas com base nos dados disponíveis no conjunto de dados de treinamento. Use sua imaginação.
# 
# **R**: Na construção de um modelo, geralmente é útil começar criando um modelo inicial simples e intuitivo com base nos resultados da análise exploratória de dados.
# Você pode sentir vontade de apresentar habilidades estatísticas mais avançadas. Por esse motivo, estamos fornecendo dados de teste no conjunto de dados ames_test para que você possa construir um modelo simples para prever os preços das casas com base nos dados disponíveis no conjunto de dados de treinamento. Use sua imaginação.
# R: No modelo de machine learning em primeiro fiz o pré-processamento dos dados mudando os tipo de dados de float para inteiro. O segundo passo eu fiz feature engineering nos dados na variável, dependente séria "Price". No terceiro passo que fiz declara as variáveis para treino e teste. A outra etapa seria o escalonamento dos dados, transformando a variável treino para média zero e variância unitária. Outra etapa treinamento do modelo 20 para treino, 80 para teste. Por último os modelos de machine learning que eu utilizei foi a regressão linear, random forest regressor, K-NN regressor, decision tree regressor na minha análise o modelo teve resultado ótimo foi o Decision Tree Regressor, o segundo Random Forest no primeiro modelo teve uma acurácia de 95.60%, o segundo teve 82.58%. E foi utilizado as métricas como RMSE, MAE, MSE, MAPE, R2, nós modelo teve resultados ótimos.
# A previsão dos imóveis com a target "Price" prevê o valor de imóvel pela idade das pessoas. Portanto nesse case tive algumas coisas que aprendi foi modelagem de dados, análise de dados, pré-processamento, engenharia de recursos dentro dos dados de ames_test. Nesse case tive uma visão para utilizar modelos de regressões.

# **Modelo machine learning 01 - Regressão linear**

# In[48]:


# Modelo regressão linear - 1
# Importação da biblioteca sklearn o modelo regressão linear

from sklearn.linear_model import LinearRegression 

# Nome do algoritmo M.L
model_linear = LinearRegression() 

# Treinamento do modelo
model_linear_fit = model_linear.fit(x_train, y_train)

# Score do modelo
model_linear_score_1 = model_linear.score(x_train, y_train)

# Previsão do modelo

model_linear_pred = model_linear.predict(x_test)
model_linear_pred


# In[49]:


# O intercepto representa o efeito médio em tendo todas as variáveis explicativas excluídas do modelo. 
# De forma mais simples, o intercepto representa o efeito médio em são iguais a zero.

model_linear.intercept_


# In[50]:


# Os coeficientes de regressão  𝛽2 ,  𝛽3  e  𝛽4  são conhecidos como coeficientes parciais de regressão ou coeficientes parciais angulares. 
# Considerando o número de variáveis explicativas de nosso modelo, seu significado seria o seguinte

model_linear.coef_


# In[51]:


# O coeficiente de determinação (R²) é uma medida resumida que diz quanto a linha de regressão ajusta-se aos dados. 
# É um valor entra 0 e 1.

print('R² = {}'.format(model_linear.score(x_train, y_train).round(2)))


# In[52]:


# Previsão do modelo 
pred = model_linear.predict(x_train)
pred2 = y_train - pred
pred2


# In[53]:


# Grafico de regressão linear

plt.figure(figsize=(18, 8))
plt.scatter(pred, y_train)
plt.plot(pred, model_linear.predict(x_train), color = "red")
plt.title("Grafico de regressão linear", fontsize = 20)
plt.xlabel("Total")
plt.ylabel("Valor dos imóveis")
plt.legend(["Preço", "Imóvel"])


# In[54]:


# Gráfico de distribuição Frequências

ax = sns.distplot(pred)
ax.figure.set_size_inches(20, 8)
ax.set_title('Distribuição de Frequências dos Resíduos', fontsize=18)
ax.set_xlabel('Internações', fontsize=14)
ax


# # 13) Métricas para o modelo de regressão linear

# - RMSE: Raiz do erro quadrático médio 
# - MAE: Erro absoluto médio  
# - MSE: Erro médio quadrático
# - MAPE: Erro Percentual Absoluto Médio
# - R2: O R-Quadrado, ou Coeficiente de Determinação, é uma métrica que visa expressar a quantidade da variança dos dados.

# In[55]:


# Importando bibliotecas verificações das métricas 

from math import sqrt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(y_test, model_linear_pred))
mae = mean_absolute_error(y_test, model_linear_pred)
mape = mean_absolute_percentage_error(y_test, model_linear_pred)
mse = mean_squared_error(y_test, model_linear_pred)
r2 = r2_score(y_test, model_linear_pred)

pd.DataFrame([rmse, mae, mse, mape, r2], ['RMSE', 'MAE', 'MSE', "MAPE",'R²'], columns=['Resultado'])


# In[56]:


# Previsão dos preços dos imóvel

prev = x_test[0:25]
model_pred = model_linear.predict(prev)[0]
print("Previsão do imóvel", model_pred)
prev


# # 14) Modelo 02 - Random Forest Regressor

# In[57]:


from sklearn.ensemble import RandomForestRegressor

model_random_forest_regressor = RandomForestRegressor(max_depth=20, random_state=0)
model_random_forest_regressor_fit = model_random_forest_regressor.fit(x_train, y_train)
model_random_forest_regressor_score = model_random_forest_regressor.score(x_train, y_train)

print("Modelo - Random forest regressor score: %.2f" % (model_random_forest_regressor_score * 100))


# In[58]:


model_random_forest_regressor_pred = model_random_forest_regressor.predict(x_test)
model_random_forest_regressor_pred


# In[59]:


# O coeficiente de determinação (R²) é uma medida resumida que diz quanto a linha de regressão ajusta-se aos dados. 
# É um valor entra 0 e 1.

print('R² = {}'.format(model_random_forest_regressor.score(x_train, y_train).round(2)))


# In[60]:


# Previsão do modelo 
pred = model_random_forest_regressor.predict(x_train)
pred2 = y_train - pred
pred2


# In[61]:


# Grafico de regressão linear

plt.figure(figsize=(18, 8))
plt.scatter(pred, y_train)
plt.plot(pred, model_random_forest_regressor.predict(x_train), color = "red")
plt.title("Grafico de regressão linear", fontsize = 20)
plt.xlabel("Total")
plt.xlabel("Total")
plt.ylabel("Valor dos imóveis")
plt.legend(["Imóvel", "Preço"])


# In[62]:


# Gráfico de distribuição Frequências

ax = sns.distplot(pred)
ax.figure.set_size_inches(20, 8)
ax.set_title('Distribuição de Frequências dos Resíduos', fontsize=18)
ax.set_xlabel('Valor', fontsize=14)
ax


# # 15) Métricas para o modelo 2 Random Forest Regressor
# 
# - RMSE: Raiz do erro quadrático médio 
# - MAE: Erro absoluto médio  
# - MSE: Erro médio quadrático
# - MAPE: Erro Percentual Absoluto Médio
# - R2: O R-Quadrado, ou Coeficiente de Determinação, é uma métrica que visa expressar a quantidade da variança dos dados.

# In[63]:


rmse = np.sqrt(mean_squared_error(y_test, model_random_forest_regressor_pred))
mae = mean_absolute_error(y_test, model_random_forest_regressor_pred)
mape = mean_absolute_percentage_error(y_test, model_random_forest_regressor_pred)
mse = mean_squared_error(y_test, model_random_forest_regressor_pred)
r2 = r2_score(y_test, model_random_forest_regressor_pred)

pd.DataFrame([rmse, mae, mse, mape, r2], ['RMSE', 'MAE', 'MSE', "MAPE",'R²'], columns=['Resultado'])


# In[64]:


# Previsão de imóvel

prev = x_test[0:25]
model_pred = model_random_forest_regressor.predict(prev)[0]
print("Previsão valor do imóvel", model_pred)
prev


# # 16) Modelo 03 - KNN Regressor

# In[65]:


from sklearn.neighbors import KNeighborsRegressor

modelo_KNN_regressor = KNeighborsRegressor(n_neighbors = 30, metric = 'euclidean')
modelo_KNN_regressor_fit = modelo_KNN_regressor.fit(x_train, y_train)
modelo_KNN_regressor_score = modelo_KNN_regressor.score(x_train, y_train)

print("Modelo - K-NN regressor score: %.2f" % (modelo_KNN_regressor_score * 100))


# In[66]:


modelo_KNN_regressor_pred = modelo_KNN_regressor.predict(x_test)
modelo_KNN_regressor_pred


# In[67]:


# O coeficiente de determinação (R²) é uma medida resumida que diz quanto a linha de regressão ajusta-se aos dados. 
# É um valor entra 0 e 1.

print('R² = {}'.format(modelo_KNN_regressor.score(x_train, y_train).round(2)))


# In[68]:


# Previsão do modelo 
pred = modelo_KNN_regressor.predict(x_train)
pred2 = y_train - pred
pred2


# In[69]:


# Grafico de regressão linear

plt.figure(figsize=(18, 8))
plt.scatter(pred, y_train)
plt.plot(pred, modelo_KNN_regressor.predict(x_train), color = "red")
plt.title("Grafico de regressão linear", fontsize = 20)
plt.xlabel("Total")
plt.xlabel("Total")
plt.ylabel("Valor dos imóveis")
plt.legend(["Preço", "Imóvel"])


# In[70]:


# Gráfico de distribuição Frequências

ax = sns.distplot(pred)
ax.figure.set_size_inches(20, 8)
ax.set_title('Distribuição de Frequências dos Resíduos', fontsize=18)
ax.set_xlabel('Valor', fontsize=14)
ax


# # 17) Métricas para o modelo 3 K-NN Regressor
# 
# - RMSE: Raiz do erro quadrático médio 
# - MAE: Erro absoluto médio  
# - MSE: Erro médio quadrático
# - MAPE: Erro Percentual Absoluto Médio
# - R2: O R-Quadrado, ou Coeficiente de Determinação, é uma métrica que visa expressar a quantidade da variança dos dados.

# In[71]:


rmse = np.sqrt(mean_squared_error(y_test, modelo_KNN_regressor_pred))
mae = mean_absolute_error(y_test, modelo_KNN_regressor_pred)
mape = mean_absolute_percentage_error(y_test, modelo_KNN_regressor_pred)
mse = mean_squared_error(y_test, modelo_KNN_regressor_pred)
r2 = r2_score(y_test, modelo_KNN_regressor_pred)

pd.DataFrame([rmse, mae, mse, mape, r2], ['RMSE', 'MAE', 'MSE', "MAPE",'R²'], columns=['Resultado'])


# In[72]:


# Previsão de imóvel

prev = x_test[0:25]
model_pred = modelo_KNN_regressor.predict(prev)[0]
print("Previsão de imóvel", model_pred)
prev


# # 18) Modelo 04 - Decision Tree Regressor

# In[73]:


from sklearn.tree import DecisionTreeRegressor

model_decision_tree_regressor = DecisionTreeRegressor(random_state = 30)
model_decision_tree_regressor_fit = model_decision_tree_regressor.fit(x_train, y_train)
model_decision_tree_regressor_score = model_decision_tree_regressor.score(x_train, y_train)

print("Modelo - Decision tree regressor score: %.2f" % (model_decision_tree_regressor_score * 100))


# In[74]:


model_decision_tree_regressor_pred = model_decision_tree_regressor.predict(x_test)
model_decision_tree_regressor_pred


# In[75]:


# O coeficiente de determinação (R²) é uma medida resumida que diz quanto a linha de regressão ajusta-se aos dados. 
# É um valor entra 0 e 1.

print('R² = {}'.format(model_decision_tree_regressor.score(x_train, y_train).round(2)))


# In[76]:


# Previsão do modelo 
pred = model_decision_tree_regressor.predict(x_train)
pred2 = y_train - pred
pred2


# In[77]:


# Grafico de regressão linear

plt.figure(figsize=(18, 8))
plt.scatter(pred, y_train)
plt.plot(pred, model_decision_tree_regressor.predict(x_train), color = "red")
plt.title("Grafico de regressão linear", fontsize = 20)
plt.xlabel("Total")
plt.xlabel("Total")
plt.ylabel("Valor dos imóveis")
plt.legend(["Preço", "Imóvel"])


# In[78]:


# Gráfico de distribuição Frequências

ax = sns.distplot(pred)
ax.figure.set_size_inches(20, 8)
ax.set_title('Distribuição de Frequências dos Resíduos', fontsize=18)
ax.set_xlabel('Valor', fontsize=14)
ax


# # 19) Métricas para o modelo 4 Decision Tree Regressor
# 
# - RMSE: Raiz do erro quadrático médio
# - MAE: Erro absoluto médio
# - MSE: Erro médio quadrático
# - MAPE: Erro Percentual Absoluto Médio
# - R2: O R-Quadrado, ou Coeficiente de Determinação, é uma métrica que visa expressar a quantidade da variança dos dados.

# In[79]:


rmse = np.sqrt(mean_squared_error(y_test, model_decision_tree_regressor_pred))
mae = mean_absolute_error(y_test, model_decision_tree_regressor_pred)
mape = mean_absolute_percentage_error(y_test, model_decision_tree_regressor_pred)
mse = mean_squared_error(y_test, model_decision_tree_regressor_pred)
r2 = r2_score(y_test, model_decision_tree_regressor_pred)

pd.DataFrame([rmse, mae, mse, mape, r2], ['RMSE', 'MAE', 'MSE', "MAPE",'R²'], columns=['Resultado'])


# In[80]:


# Previsão de imóvel

prev = x_test[0:25]
model_pred = model_decision_tree_regressor.predict(prev)[0]
print("Previsão de imóvel", model_pred)
prev


# # 20) Resultados final dos modelos

# In[81]:


# Exibindo um comparativo dos modelos de regressão linear

modelos = pd.DataFrame({
    
    "Modelos" :["Modelo Regressão Linear", 
                "Modelo Random Forest", 
                "Modelo K-NN Regressor",
                "Modelo Decision Tree Regressor"],

    "Acurácia" :[model_linear_score_1, 
                 model_random_forest_regressor_score, 
                 modelo_KNN_regressor_score,
                 model_decision_tree_regressor_score]})

modelos.sort_values(by = "Acurácia", ascending = False)


# # 21) Salvando modelo de ML

# In[82]:


# Salvando modelo de regressão linear

import pickle

with open('model_linear_pred.pkl', 'wb') as file:
    pickle.dump(model_linear_pred, file)
    
with open('model_random_forest_regressor_pred.pkl', 'wb') as file:
    pickle.dump(model_random_forest_regressor_pred, file)
    
with open('modelo_KNN_regressor_pred.pkl', 'wb') as file:
    pickle.dump(modelo_KNN_regressor_pred, file)
    
with open('model_decision_tree_regressor_pred.pkl', 'wb') as file:
    pickle.dump(model_decision_tree_regressor_pred, file)


# # 22) Conclusão do modelo machine learning

# Pela análise dos modelos, modelo 1 decision tree regressor, e o segundo modelo random forest teve melhor resultado que os demais, atigindo uma acurácia de 95.53% para decision tree, random forest de 82.25% ou seja capaz de acertar as previsões de valor do imóvel. De acordo com análise realizada.  

# # 23) Referência 

# **Data Processing**
# 
# https://towardsdatascience.com/introduction-to-data-preprocessing-in-machine-learning-a9fa83a5dc9d
# 
# **Feature Engineering**
# 
# https://ateliware.com/blog/feature-engineering
# 
# **Escalonamento**
# 
# https://www.brutalk.com/en/news/brutalk-blog/view/como-usar-o-escalonamento-de-dados-melhorar-a-estabilidade-e-o-desempenho-do-modelo-de-aprendizado-profundo-6046ffa588320

# In[ ]:




