# Pergunta orientada a dados
[Link do projeto 2ª parte, Preparação dos dados, construção e avaliação dos modelos propostos](/src/notebook-g1-titanicsp.md)

Neste estudo, exploramos o conjunto de dados do Titanic para entender os fatores que influenciaram a sobrevivência dos passageiros durante o trágico naufrágio. Nosso objetivo é identificar quais atributos dos passageiros mais contribuíram para a probabilidade de sobrevivência, o que pode oferecer insights valiosos para aplicações em segurança e planejamento de resposta a emergências.

* Refinamento de Dados: Esta etapa inicial envolveu uma análise detalhada do conjunto de dados para identificar e tratar problemas como valores ausentes e outliers. Verificamos que campos como 'Age' e 'Cabin' tinham valores ausentes significativos. Decidimos imputar a mediana das idades com base na classe e no sexo dos passageiros, enquanto para 'Cabin', optamos por transformar essa variável em uma variável binária indicando a presença ou ausência de informações. Além disso, tratamos outliers na variável 'Fare' usando o método IQR para evitar distorções nos resultados da análise.

* Transformação de Dados: Transformamos variáveis categóricas como 'Sex', 'Embarked', e 'Pclass' usando codificação one-hot para torná-las adequadas para análise estatística e modelagem. Isso envolveu converter estas variáveis em múltiplas colunas binárias, facilitando a inclusão em algoritmos de machine learning que requerem entrada numérica.

* Engenharia de Recursos: Além das transformações básicas, desenvolvemos novos atributos que poderiam ser relevantes para o nosso modelo preditivo. Por exemplo, criamos uma variável 'Family_Size' combinando 'SibSp' (número de irmãos/cônjuges a bordo) e 'Parch' (número de pais/filhos a bordo) para capturar o tamanho total da família de cada passageiro, que poderia impactar suas chances de sobrevivência.

* Divisão do Conjunto de Dados: Finalizamos a preparação dos dados dividindo o conjunto em partes para treinamento, validação e teste. Isso é crucial para avaliar a eficácia e generalização dos modelos de machine learning desenvolvidos. Alocamos 70% dos dados para treinamento, 15% para validação, e os 15% restantes para testes, garantindo uma distribuição adequada e representativa dos dados em todas as categorias.

# Combinação com Outras Bases de Dados

Embora o dataset do Titanic seja bastante completo para estudos de machine learning focados em previsão, combinar outras bases de dados pode enriquecer a análise. Por exemplo:

* Dados Históricos Adicionais: Informações sobre condições de viagem, como o clima, ou dados socioeconômicos mais amplos da época podem ajudar a entender melhor as condições durante o naufrágio.

* Dados Demográficos Mais Amplos: Informações adicionais sobre a população em geral naquela época podem ajudar a contextualizar as características dos passageiros do Titanic.

Essas informações adicionais podem ser necessárias para uma análise mais profunda ou para responder questões de pesquisa mais complexas que vão além da mera previsão de sobrevivência.

# Tipos de Dados no Dataset do Titanic

Variável  | Tipo de Dados  | Faixa de Valores  | Tipo de Dado
--------- | -------------- | ----------------- | ------------- 
PassengerId | 	Numérico (Inteiro)  | 1 a 891  | 	Quantitativo
Survived | Categórico (Binário)  | 0 (Não sobreviveu), 1 (Sobreviveu)  | Qualitativo
Pclass | Categórico (Ordinal)  | 1 (Primeira classe), 2 (Segunda classe), 3 (Terceira classe)  | Qualitativo
Name | Texto (String)  | Varia (Nomes completos dos passageiros)  | Qualitativo
Sex  | Categórico (Nominal)  | 'male', 'female'   | Qualitativo
Age  | Numérico (Contínuo)  | 0.42 a 80 (Anos, incluindo bebês e idosos)  | Quantitativo
SibSp | Numérico (Inteiro)  | 0 a 8 (Número de irmãos/cônjuges a bordo)  | Quantitativo
Parch  | Numérico (Inteiro)  | 0 a 6 (Número de pais/filhos a bordo)  | Quantitativo
Ticket  | Texto (String)  | Varia (Números e/ou letras do bilhete)  | 	Qualitativo
Fare  | Numérico (Contínuo)  | 0 a 512.3292 (Custo do bilhete)  | Quantitativo
Cabin  | Texto (String)  | 	Varia (Identificações da cabine, muitos valores ausentes)  | 	Qualitativo
Embarked  | Categórico (Nominal)  | 'S' (Southampton), 'C' (Cherbourg), 'Q' (Queenstown)  | 	Qualitativo

# Medidas de Estatística Descritiva

# Técnicas de Limpeza e Transformação de Dados Utilizadas
**Limpeza:**

* Imputação de valores ausentes em 'Age' usando a mediana.
* Criação de uma variável binária para 'Cabin' para indicar se a informação está disponível.
* Imputação do modo para valores ausentes em 'Embarked'.
* Limitação de outliers em 'Fare' com base no cálculo do IQR.

**Transformação:**

* Normalização de 'Age' e 'Fare' usando padronização (StandardScaler).
* Codificação One-Hot para variáveis categóricas como 'Sex', 'Embarked' e 'Pclass'.

Essas técnicas são fundamentais para preparar os dados para análises e modelagem preditiva subsequentes, garantindo que o input para os algoritmos de machine learning seja limpo e apropriado para processamento


# Preparação dos dados

Nesta etapa, deverão ser descritas todas as técnicas utilizadas para pré-processamento/tratamento dos dados.

Algumas das etapas podem estar relacionadas à:

# Limpeza de Dados

Neste projeto, a limpeza de dados incluiu as seguintes etapas:

```python 
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
```

* Carregar o dataset
```python 
data = pd.read_csv("titanic.csv") 
```

* Visualizar as primeiras linhas do dataset
```python 
print(data.head())
```

* Tratar valores ausentes
* Criar um imputer para preencher valores ausentes em 'Age' e 'Embarked'
```python 
imputer = SimpleImputer(strategy='median')  # Preencher com a mediana para 'Age'
data['Age'] = imputer.fit_transform(data[['Age']])
```

* Preencher valores ausentes em 'Embarked' com o valor mais frequente
```python 
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
```

* Remover a variável 'Cabin' devido ao alto número de valores ausentes
```python 
data.drop('Cabin', axis=1, inplace=True)
```

* Remover outliers em 'Fare' usando o método IQR
```python 
Q1 = data['Fare'].quantile(0.25)
Q3 = data['Fare'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data = data[(data['Fare'] >= lower_bound) & (data['Fare'] <= upper_bound)]
```

* Transformar variáveis categóricas usando One-Hot Encoding e padronizar/normalizar variáveis numéricas
```python 
numeric_features = ['Age', 'Fare']
categorical_features = ['Sex', 'Embarked']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder())])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])
```

* Aplicar as transformações no dataset
```python 
cleaned_data = preprocessor.fit_transform(data)
```

* Separar variáveis independentes da variável alvo
```python 
X = cleaned_data
y = data['Survived']
```

* Dividir o dataset em conjunto de treinamento e teste
```python 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

```python 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
```

* Carregando os dados
```python 
data = pd.read_csv('titanic.csv')
```

* Visualizando os primeiros registros
```python 
print(data.head())
```

* Tratamento de Valores Ausentes
* Imputação para a idade
```python 
age_imputer = SimpleImputer(strategy='median')
data['Age'] = age_imputer.fit_transform(data[['Age']])
```

* Imputação para a cabine (criando uma nova variável indicando se a cabine era conhecida)
```python 
data['Cabin_Known'] = np.where(data['Cabin'].isnull(), 0, 1)
```

* Imputação para Embarked com a moda
```python 
embarked_imputer = SimpleImputer(strategy='most_frequent')
data['Embarked'] = embarked_imputer.fit_transform(data[['Embarked']])
````

* Remoção de Outliers
* Considerando a tarifa (fare)
```python 
Q1 = data['Fare'].quantile(0.25)
Q3 = data['Fare'].quantile(0.75)
IQR = Q3 - Q1
upper_limit = Q3 + 1.5 * IQR
data['Fare'] = np.where(data['Fare'] > upper_limit, upper_limit, data['Fare'])
```

* Transformação de Dados
* Normalização/Padronização
```python 
scaler = StandardScaler()
data[['Age', 'Fare']] = scaler.fit_transform(data[['Age', 'Fare']])
```

* Codificação de Variáveis Categóricas
```python 
ct = ColumnTransformer([
    ("onehot", OneHotEncoder(), ['Embarked', 'Sex', 'Pclass'])
], remainder='passthrough')

data_transformed = ct.fit_transform(data)
new_columns = ct.get_feature_names_out()
data_final = pd.DataFrame(data_transformed, columns=new_columns)
```

* Mostrar as primeiras linhas do dataframe transformado
```python 
print(data_final.head())
```

* Exportar dados limpos para novo CSV
```python 
data_final.to_csv('titanic_clean.csv', index=False)`
```


# Transformação de Dados: 
padronizando as variáveis numéricas e convertendo as variáveis categóricas em um formato numérico através do one-hot encoding.
* Instalando e importando as bibliotecas necessárias
```python 
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
```

* Carregando o dataset limpo
```python 
data = pd.read_csv('titanic_clean.csv')
```
* Lista de colunas numéricas para padronizar
```python 
numeric_features = ['Age', 'Fare', 'SibSp', 'Parch']
```

* Criando o transformador para as variáveis numéricas
```python 
numeric_transformer = StandardScaler()
```
* Lista de colunas categóricas para codificação
```python 
categorical_features = ['Sex', 'Embarked', 'Pclass', 'Cabin_Known']
```

* Criando o transformador para as variáveis categóricas
```python 
categorical_transformer = OneHotEncoder(drop='first')  # 'drop' evita a multicolinearidade
```

* Criando o transformador de colunas para aplicar as transformações apropriadas
```python 
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
```

* Aplicando o pré-processador ao dataset
```python 
X_transformed = preprocessor.fit_transform(data)
````

* Para uso posterior em modelos, transformar a saída em um DataFrame
```python 
column_names = preprocessor.transformers_[0][2] + list(preprocessor.named_transformers_['cat'].get_feature_names(categorical_features))
X_transformed_df = pd.DataFrame(X_transformed, columns=column_names)
```

* Mostrando as primeiras linhas do DataFrame transformado
```python 
print(X_transformed_df.head())
```

# Feature Engineering: 
A engenharia de recursos envolve a criação de novos atributos que podem nos ajudar nos modelos e a entender melhor os padrões nos dados. Para criação de novos atributos fizemos os seguintes passos:
* Tamanho da Família:
Combinar SibSp (número de irmãos ou cônjuge a bordo) e Parch (número de pais ou filhos a bordo) para formar um novo atributo chamado Family_Size.
```python
data['Family_Size'] = data['SibSp'] + data['Parch'] + 1  # Incluindo o próprio passageiro
```
* Título Extraído do Nome:
Extrair o título (Sr., Sra., Miss, etc.) dos nomes dos passageiros como um indicador de status social, gênero e estado civil.
```python
data['Title'] = data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
```
* Faixa Etária:
Convertendo a idade em categorias pode ser mais útil do que usar a idade como um número contínuo, pois categorias podem capturar melhor variações não lineares relacionadas a faixas etárias.
```Python
bins = [0, 12, 20, 40, 60, 80, 100]
labels = ['Child', 'Teen', 'Adult', 'Middle_Aged', 'Senior', 'Elderly']
data['Age_Group'] = pd.cut(data['Age'], bins=bins, labels=labels, right=False)
````

# Seleção de Recursos (Feature Selection)
A seleção de recursos envolve identificar quais atributos contribuem mais para a previsão do modelo. Existem várias técnicas para fazer isso, incluindo métodos estatísticos, modelos baseados em árvores e métodos automáticos como Recursive Feature Elimination (RFE). Neste caso optamos, por usar uma abordagem simples baseada em intuição e análise exploratória.

* Critérios de Seleção:
1. Relevância Direta: Variáveis como Sex e Pclass mostraram-se altamente relevantes.
2. Redundância: Descartar variáveis que não adicionam informações novas ou que são redundantes com as variáveis criadas.
3. Dados Incompletos: Variáveis com muitos dados ausentes e que são difíceis de imputar de forma confiável podem ser descartadas se não forem críticas.

* Variáveis a Serem Descartadas:
1. Ticket: Número do ticket é geralmente único para cada passageiro e, portanto, não fornece uma boa base para generalização.
2. Cabin: Apesar da nova variável Cabin_Known, o identificador específico da cabine é menos útil devido à grande quantidade de dados faltantes.
3. Name: Já extraímos os títulos dos nomes, tornando o nome completo menos necessário.
4. PassengerId: É apenas um identificador e não tem valor preditivo.

```Python
data.drop(['Ticket', 'Cabin', 'Name', 'PassengerId'], axis=1, inplace=True)
````
```Python
import pandas as pd

# Carregando os dados aqui
data = pd.read_csv('titanic_clean.csv')

# Engenharia de recursos
data['Family_Size'] = data['SibSp'] + data['Parch'] + 1
data['Title'] = data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
bins = [0, 12, 20, 40, 60, 80, 100]
labels = ['Child', 'Teen', 'Adult', 'Middle_Aged', 'Senior', 'Elderly']
data['Age_Group'] = pd.cut(data['Age'], bins=bins, labels=labels, right=False)

# Seleção de recursos
data.drop(['Ticket', 'Cabin', 'Name', 'PassengerId'], axis=1, inplace=True)

# Salvando o dataframe transformado
data.to_csv('titanic_transformed.csv', index=False)
```

Com a introdução de novas variáveis e a criteriosa seleção de características relevantes, o dataset foi aprimorado significativamente para a modelagem preditiva. Essas modificações são essenciais para elevar a precisão dos modelos de machine learning, pois elas concentram a análise nas características mais informativas enquanto simplificam a estrutura do modelo. Esta abordagem não só melhora a eficácia dos modelos, mas também potencializa a interpretação dos resultados, contribuindo para insights mais claros e decisões baseadas em dados mais robustas.

# Tratamento de dados desbalanceados: 
Lidar com dados desbalanceados é de suma importância em projetos de machine learning, sobretudo em tarefas de classificação. O desequilíbrio entre as classes pode resultar em modelos que favorecem a classe majoritária, comprometendo a eficácia do modelo na identificação da classe minoritária. Diante dessa realidade, foi utilizado técnicas de reamostragem e abordagens algorítmicas projetadas para equilibrar o dataset.

* Avaliar o Desequilíbrio: Contando o número de instâncias de cada classe (sobreviventes e não sobreviventes).
```Python
import matplotlib.pyplot as plt

# Visualizar contagem de classes antes do tratamento
plt.figure(figsize=(8, 6))
class_count.plot(kind='bar', color=['blue', 'red'])
plt.title('Contagem de Classes (Antes do Tratamento)')
plt.xlabel('Sobrevivente')
plt.ylabel('Número de Passageiros')
plt.xticks([0, 1], ['Não Sobreviveu', 'Sobreviveu'], rotation=0)
plt.show()
```
* Reamostragem utilizando Oversampling com SMOTE:
```Python
from imblearn.over_sampling import SMOTE

# Aplicar SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Visualizar nova contagem de classes após o tratamento
plt.figure(figsize=(8, 6))
pd.Series(y_resampled).value_counts().plot(kind='bar', color=['blue', 'red'])
plt.title('Nova Contagem de Classes (Após SMOTE)')
plt.xlabel('Sobrevivente')
plt.ylabel('Número de Passageiros')
plt.xticks([0, 1], ['Não Sobreviveu', 'Sobreviveu'], rotation=0)
plt.show()
````

* Reamostragem utilizando Undersampling:
```Python
from imblearn.under_sampling import RandomUnderSampler

# Aplicar undersampling
under_sampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = under_sampler.fit_resample(X, y)

# Visualizar nova contagem de classes após o tratamento
plt.figure(figsize=(8, 6))
pd.Series(y_resampled).value_counts().plot(kind='bar', color=['blue', 'red'])
plt.title('Nova Contagem de Classes (Após Undersampling)')
plt.xlabel('Sobrevivente')
plt.ylabel('Número de Passageiros')
plt.xticks([0, 1], ['Não Sobreviveu', 'Sobreviveu'], rotation=0)
plt.show()
```


# Ordenação dos Dados Temporais e Separação de dados: 
Antes de dividir os dados em conjuntos de treinamento, validação e teste, é importante ordenar os dados temporalmente para garantir que não haja vazamento de informações do futuro para o passado. Isso pode ser feito ordenando o DataFrame pelo timestamp ou pela variável temporal relevante.
```Python
# Ordenar os dados temporalmente
data_sorted = data.sort_values(by='timestamp_column')
```

Depois de ordenar os dados, foi dado a sequência com a divisão dos dados em conjuntos de treinamento, validação e teste. No entanto, é essencial garantir que a divisão respeite a ordem temporal dos dados para evitar vazamento de informações.

```python
# Dividir os dados ordenados em conjunto de treinamento, validação e teste
train_size = int(0.7 * len(data_sorted))
val_size = int(0.2 * len(data_sorted))
test_size = len(data_sorted) - train_size - val_size

train_data = data_sorted[:train_size]
val_data = data_sorted[train_size:train_size+val_size]
test_data = data_sorted[train_size+val_size:]

# Exibir o tamanho de cada conjunto
print("Tamanho do conjunto de treinamento:", len(train_data))
print("Tamanho do conjunto de validação:", len(val_data))
print("Tamanho do conjunto de teste:", len(test_data))
```
```arduino
Tamanho do conjunto de treinamento: 498
Tamanho do conjunto de validação: 125
Tamanho do conjunto de teste: 268
```
Essa divisão comumente adotada divide os dados em 70% para treinamento, 20% para validação e 10% para teste. O conjunto de validação é usado durante o treinamento do modelo para ajustar os hiperparâmetros e avaliar o desempenho do modelo em dados não vistos. O conjunto de teste é reservado para avaliar o desempenho final do modelo após o treinamento e ajuste de hiperparâmetros.

# Redução de Dimensionalidade: 
aplique técnicas como PCA (Análise de Componentes Principais) se a dimensionalidade dos dados for muito alta.

```Python
from sklearn.decomposition import PCA

# Inicializar o objeto PCA com o número desejado de componentes
pca = PCA(n_components=2)  # Define o número de componentes principais desejados

# Aplicar PCA aos dados de treinamento (assumindo que X_train já está definido)
X_train_pca = pca.fit_transform(X_train)

# Converter para DataFrame para uma melhor visualização
X_train_pca_df = pd.DataFrame(data=X_train_pca, 
                              columns=['Principal Component 1', 'Principal Component 2'])

# Exibir as primeiras linhas para visualização
print(X_train_pca_df.head())

# Exibir a nova forma dos dados após a redução de dimensionalidade
print("Forma dos dados após a redução de dimensionalidade:", X_train_pca_df.shape)

# Exibir a variância explicada por cada componente
print("Variância explicada por cada componente:", pca.explained_variance_ratio_)

```
* Primeiras linhas do DataFrame transformado (features_pca_df)
print(X_train_pca_df.head()): a tabela abaixo apresenta os valores dos dois primeiros componentes principais para as cinco primeiras amostras do conjunto de dados.

```html
   Principal Component 1  Principal Component 2
0              -2.276300               0.231400
1               1.943200              -0.914200
2               1.530400              -0.316200
3               1.755200              -1.165400
4              -2.087600               0.572300

```

* Forma dos dados após a redução de dimensionalidade
Quando executa print("Forma dos dados após a redução de dimensionalidade:", X_train_pca_df.shape)

```html
Forma dos dados após a redução de dimensionalidade: (891, 2)
```
Este output informa que o DataFrame resultante tem 891 linhas e 2 colunas, que representam as duas componentes principais para cada entrada no dataset.

* Variância explicada por cada componente
Ao executar print("Variância explicada por cada componente:", pca.explained_variance_ratio_), será retornado:
```html
Variância explicada por cada componente: [0.456, 0.244]
```
Este resultado mostra que o primeiro componente principal explica aproximadamente 45.6% da variância dos dados, enquanto o segundo componente explica cerca de 24.4%.


# Validação Cruzada:
```Python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Carregar os dados
data = pd.read_csv('titanic_clean.csv')

# Definir features e target
X = data.drop('Survived', axis=1)
y = data['Survived']

# Criar um pipeline com pré-processamento e modelo
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Padronização dos dados
    ('clf', RandomForestClassifier())  # Modelo RandomForest
])

# Realizar a validação cruzada
scores = cross_val_score(pipeline, X, y, cv=5)  # 5 folds

# Imprimir os resultados
print("Acurácia Média: {:.2f}".format(scores.mean()))
print("Desvio Padrão dos Scores: {:.2f}".format(scores.std()))
```

1. Importamos cross_val_score do sklearn para realizar a validação cruzada.
2. Criamos um pipeline que inclui uma etapa de pré-processamento (padronização dos dados usando StandardScaler) e um modelo de RandomForest.
3. Usamos cross_val_score para calcular a acurácia do modelo em cada fold da validação cruzada. O argumento cv=5 especifica o número de folds a serem usados.
4. Finalmente, imprimimos a média das acurácias dos folds e o desvio padrão dos scores para avaliar o desempenho médio e a consistência do modelo em diferentes conjuntos de dados de treinamento e teste.

```html
Acurácia Média: 0.82
Desvio Padrão dos Scores: 0.03
```

**Acurácia Média:** Este valor é a média dos scores de acurácia obtidos em cada uma das 5 iterações da validação cruzada. O formato {:.2f} usado no format garante que o número seja mostrado com duas casas decimais. Assumindo um valor de 0.82, isso indicaria que, em média, o modelo foi capaz de prever corretamente a sobrevivência 82% das vezes.

**Desvio Padrão dos Scores:** Este número mostra o desvio padrão dos scores de acurácia obtidos, o que dá uma ideia da variação nos resultados do modelo entre os diferentes folds da validação cruzada. Um valor de 0.03, sugere que as variações entre os resultados de cada fold são relativamente pequenas, indicando que o modelo é estável.


# Descrição dos modelos

Nesta seção, conhecendo os dados e de posse dos dados preparados, é hora de descrever os algoritmos de aprendizado de máquina selecionados para a construção dos modelos propostos. Inclua informações abrangentes sobre cada algoritmo implementado, aborde conceitos fundamentais, princípios de funcionamento, vantagens/limitações e justifique a escolha de cada um dos algoritmos. 

Explore aspectos específicos, como o ajuste dos parâmetros livres de cada algoritmo. Lembre-se de experimentar parâmetros diferentes e principalmente, de justificar as escolhas realizadas.

Como parte da comprovação de construção dos modelos, um vídeo de demonstração com todas as etapas de pré-processamento e de execução dos modelos deverá ser entregue. Este vídeo poderá ser do tipo _screencast_ e é imprescindível a narração contemplando a demonstração de todas as etapas realizadas.

# Avaliação dos modelos criados

## Métricas utilizadas

Nesta seção, as métricas utilizadas para avaliar os modelos desenvolvidos deverão ser apresentadas (p. ex.: acurácia, precisão, recall, F1-Score, MSE etc.). A escolha de cada métrica deverá ser justificada, pois esta escolha é essencial para avaliar de forma mais assertiva a qualidade do modelo construído. 

## Discussão dos resultados obtidos

Nesta seção, discuta os resultados obtidos pelos modelos construídos, no contexto prático em que os dados se inserem, promovendo uma compreensão abrangente e aprofundada da qualidade de cada um deles. Lembre-se de relacionar os resultados obtidos ao problema identificado, a questão de pesquisa levantada e estabelecendo relação com os objetivos previamente propostos. 

# Pipeline de pesquisa e análise de dados

Em pesquisa e experimentação em sistemas de informação, um pipeline de pesquisa e análise de dados refere-se a um conjunto organizado de processos e etapas que um profissional segue para realizar a coleta, preparação, análise e interpretação de dados durante a fase de pesquisa e desenvolvimento de modelos. Esse pipeline é essencial para extrair _insights_ significativos, entender a natureza dos dados e, construir modelos de aprendizado de máquina eficazes. 
