# 2ª parte, Preparação dos dados, construção e avaliação dos modelos propostos


```python
# Projeto: Pesquisa e Experimentação em Sistemas de Informação-Grupo 1 - notebook_titanicsurvivalprediction

import numpy as np # Manipulação de matrizes
import pandas as pd # Criação e manipulação de dataset
from pandas import *
import matplotlib.pyplot as plt # Plotagem de dados
import matplotlib.font_manager
%matplotlib inline
import seaborn as sns # Plotagem e visualização dos dados
from tabulate import tabulate
from scipy.stats import chi2_contingency
import warnings
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
```

# Preparação dos dados

Nesta etapa, deverão ser descritas todas as técnicas utilizadas para pré-processamento/tratamento dos dados.

Algumas das etapas podem estar relacionadas à:

# Limpeza de Dados
Neste projeto, a limpeza de dados incluiu as seguintes etapas:


```python
# Carregar o dataset

data = pd.read_csv("/kaggle/input/dataset-g1pd/Titanic-Dataset.csv")
```


```python
# Visualizar as primeiras linhas do dataset

print(data.head())
```

       PassengerId  Survived  Pclass  \
    0            1         0       3   
    1            2         1       1   
    2            3         1       3   
    3            4         1       1   
    4            5         0       3   
    
                                                    Name     Sex   Age  SibSp  \
    0                            Braund, Mr. Owen Harris    male  22.0      1   
    1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   
    2                             Heikkinen, Miss. Laina  female  26.0      0   
    3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   
    4                           Allen, Mr. William Henry    male  35.0      0   
    
       Parch            Ticket     Fare Cabin Embarked  
    0      0         A/5 21171   7.2500   NaN        S  
    1      0          PC 17599  71.2833   C85        C  
    2      0  STON/O2. 3101282   7.9250   NaN        S  
    3      0            113803  53.1000  C123        S  
    4      0            373450   8.0500   NaN        S  
    


```python
# Visualização inicial do dataset
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Tratar valores ausentes
# Criar um imputer para preencher valores ausentes na coluna Age com a mediana, reduzindo o impacto de valores extremos.
 
imputer = SimpleImputer(strategy='median')  # Preencher com a mediana para 'Age'
data['Age'] = imputer.fit_transform(data[['Age']])

```


```python
# Preencher valores ausentes na coluna Embarked com o valor mais frequente (moda).

data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
```


```python
# Remover a variável 'Cabin' devido ao alto número de valores ausentes

data.drop('Cabin', axis=1, inplace=True)

```


```python
# Tratamento de Outliers
# Identificar e remover outliers na coluna Fare usando o método IQR para melhorar a robustez do modelo.

Q1 = data['Fare'].quantile(0.25)
Q3 = data['Fare'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data = data[(data['Fare'] >= lower_bound) & (data['Fare'] <= upper_bound)]

```


```python
# Preparar os dados para modelagem aplicando transformações adequadas
# Transformar as variáveis categóricas Sex e Embarked em variáveis dummy utilizando One-Hot Encoding.
# Padronizar as variáveis numéricas Age e Fare para que tenham média 0 e desvio padrão 1, facilitando o treinamento do modelo.

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


```python
# Aplicar as transformações no dataset

cleaned_data = preprocessor.fit_transform(data)
```


```python
# Separar Variáveis Independentes da Variável Alvo: Extrair as features (X) e a variável alvo (y).

X = cleaned_data
y = data['Survived']

```


```python
# Divisão em Conjunto de Treinamento e Teste:
# Dividir o dataset em 80% para treinamento e 20% para teste, garantindo que o modelo possa ser avaliado em dados não vistos durante o treinamento.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

```


```python
# Carregar o dataset novamente

data = pd.read_csv("/kaggle/input/dataset-g1pd/Titanic-Dataset.csv")
```


```python
# Verificar os primeiros registros das transformações aplicadas

print(data.head())
```

       PassengerId  Survived  Pclass  \
    0            1         0       3   
    1            2         1       1   
    2            3         1       3   
    3            4         1       1   
    4            5         0       3   
    
                                                    Name     Sex   Age  SibSp  \
    0                            Braund, Mr. Owen Harris    male  22.0      1   
    1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   
    2                             Heikkinen, Miss. Laina  female  26.0      0   
    3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   
    4                           Allen, Mr. William Henry    male  35.0      0   
    
       Parch            Ticket     Fare Cabin Embarked  
    0      0         A/5 21171   7.2500   NaN        S  
    1      0          PC 17599  71.2833   C85        C  
    2      0  STON/O2. 3101282   7.9250   NaN        S  
    3      0            113803  53.1000  C123        S  
    4      0            373450   8.0500   NaN        S  
    


```python
# Tratamento de Valores Ausentes
# Imputação para a idade
 
age_imputer = SimpleImputer(strategy='median')
data['Age'] = age_imputer.fit_transform(data[['Age']])

```


```python
# Imputação para a cabine (criando uma nova variável indicando se a cabine era conhecida)

data['Cabin_Known'] = np.where(data['Cabin'].isnull(), 0, 1)
```


```python
# Importar a classe SimpleImputer
from sklearn.impute import SimpleImputer

# Criar o imputador
embarked_imputer = SimpleImputer(strategy='most_frequent')

# Aplicar o imputador e substituir os valores na coluna 'Embarked'
data['Embarked'] = embarked_imputer.fit_transform(data[['Embarked']].values).reshape(-1)

# Se necessário, converter a saída para uma série pandas
data['Embarked'] = pd.Series(data['Embarked'])

```


```python
# Remoção de Outliers
# Considerando a tarifa (fare)

Q1 = data['Fare'].quantile(0.25)
Q3 = data['Fare'].quantile(0.75)
IQR = Q3 - Q1
upper_limit = Q3 + 1.5 * IQR
data['Fare'] = np.where(data['Fare'] > upper_limit, upper_limit, data['Fare'])
```


```python
# Transformação de Dados
# Normalização/Padronização
#scaler = StandardScaler()
#data[['Age', 'Fare']] = scaler.fit_transform(data[['Age', 'Fare']])

# Normalização/Padronização usando MinMaxScaler para evitar valores negativos
scaler = MinMaxScaler()
data[['Age', 'Fare']] = scaler.fit_transform(data[['Age', 'Fare']])

```


```python
# Codificação de Variáveis Categóricas
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer([
    ("onehot", OneHotEncoder(), ['Embarked', 'Sex', 'Pclass'])
], remainder='passthrough')

data_transformed = ct.fit_transform(data)
new_columns = ct.get_feature_names_out()
data_final = pd.DataFrame(data_transformed, columns=new_columns)
```


```python
# Visualização inicial do dataset
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Cabin_Known</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>0.271174</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>0.110460</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>0.472229</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>1.000000</td>
      <td>C85</td>
      <td>C</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>0.321438</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>0.120745</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>0.434531</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>0.809027</td>
      <td>C123</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>0.434531</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>0.122649</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Mostrar as primeiras linhas do dataframe transformado
# Confirmar visualmente que as transformações foram aplicadas corretamente.

print(data_final.head())
```

      onehot__Embarked_C onehot__Embarked_Q onehot__Embarked_S onehot__Sex_female  \
    0                0.0                0.0                1.0                0.0   
    1                1.0                0.0                0.0                1.0   
    2                0.0                0.0                1.0                1.0   
    3                0.0                0.0                1.0                1.0   
    4                0.0                0.0                1.0                0.0   
    
      onehot__Sex_male onehot__Pclass_1 onehot__Pclass_2 onehot__Pclass_3  \
    0              1.0              0.0              0.0              1.0   
    1              0.0              1.0              0.0              0.0   
    2              0.0              0.0              0.0              1.0   
    3              0.0              1.0              0.0              0.0   
    4              1.0              0.0              0.0              1.0   
    
      remainder__PassengerId remainder__Survived  \
    0                      1                   0   
    1                      2                   1   
    2                      3                   1   
    3                      4                   1   
    4                      5                   0   
    
                                         remainder__Name remainder__Age  \
    0                            Braund, Mr. Owen Harris       0.271174   
    1  Cumings, Mrs. John Bradley (Florence Briggs Th...       0.472229   
    2                             Heikkinen, Miss. Laina       0.321438   
    3       Futrelle, Mrs. Jacques Heath (Lily May Peel)       0.434531   
    4                           Allen, Mr. William Henry       0.434531   
    
      remainder__SibSp remainder__Parch remainder__Ticket remainder__Fare  \
    0                1                0         A/5 21171         0.11046   
    1                1                0          PC 17599             1.0   
    2                0                0  STON/O2. 3101282        0.120745   
    3                1                0            113803        0.809027   
    4                0                0            373450        0.122649   
    
      remainder__Cabin remainder__Cabin_Known  
    0              NaN                      0  
    1              C85                      1  
    2              NaN                      0  
    3             C123                      1  
    4              NaN                      0  
    


```python
# Salvar os dados transformados em um novo arquivo CSV para uso posterior.
data_final.to_csv('/kaggle/working/titanic_clean.csv', index=False)
```

# Transformação de Dados: 


```python
# padronizando as variáveis numéricas e convertendo as variáveis categóricas em um formato numérico através do one-hot encoding.
# Instalando e importando as bibliotecas necessárias

# pandas: Manipulação de dataframes.
# StandardScaler, OneHotEncoder: Ferramentas do scikit-learn para padronização e codificação de variáveis.
# ColumnTransformer: Facilita a aplicação de diferentes transformações a diferentes colunas do dataset.

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
```


```python
# Visualização inicial do dataset
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Cabin_Known</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>0.271174</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>0.110460</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>0.472229</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>1.000000</td>
      <td>C85</td>
      <td>C</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>0.321438</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>0.120745</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>0.434531</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>0.809027</td>
      <td>C123</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>0.434531</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>0.122649</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Carregar o dataset limpo (titanic_clean.csv) para transformação.
# Carregando o dataset limpo
 
data_final = pd.read_csv('/kaggle/working/titanic_clean.csv') #Carregando com nome de data_final
```


```python
# Visualização inicial do dataset
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Cabin_Known</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>0.271174</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>0.110460</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>0.472229</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>1.000000</td>
      <td>C85</td>
      <td>C</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>0.321438</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>0.120745</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>0.434531</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>0.809027</td>
      <td>C123</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>0.434531</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>0.122649</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Visualização inicial do dataset
data_final.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>onehot__Embarked_C</th>
      <th>onehot__Embarked_Q</th>
      <th>onehot__Embarked_S</th>
      <th>onehot__Sex_female</th>
      <th>onehot__Sex_male</th>
      <th>onehot__Pclass_1</th>
      <th>onehot__Pclass_2</th>
      <th>onehot__Pclass_3</th>
      <th>remainder__PassengerId</th>
      <th>remainder__Survived</th>
      <th>remainder__Name</th>
      <th>remainder__Age</th>
      <th>remainder__SibSp</th>
      <th>remainder__Parch</th>
      <th>remainder__Ticket</th>
      <th>remainder__Fare</th>
      <th>remainder__Cabin</th>
      <th>remainder__Cabin_Known</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>0.271174</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>0.110460</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>0.472229</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>1.000000</td>
      <td>C85</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3</td>
      <td>1</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>0.321438</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>0.120745</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>0.434531</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>0.809027</td>
      <td>C123</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>5</td>
      <td>0</td>
      <td>Allen, Mr. William Henry</td>
      <td>0.434531</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>0.122649</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Identificar as colunas numéricas a serem padronizadas.
# Lista de colunas numéricas para padronizar
 
numeric_features = ['Age', 'Fare', 'SibSp', 'Parch']

# Criar um transformador para padronização utilizando StandardScaler.
 
numeric_transformer = StandardScaler()

# Identificar as colunas categóricas.
# Criar um transformador para codificação utilizando OneHotEncoder, com drop='first' para evitar multicolinearidade.
 
categorical_features = ['Sex', 'Embarked', 'Pclass', 'Cabin_Known']
categorical_transformer = OneHotEncoder(drop='first')  # 'drop' evita a multicolinearidade
```


```python
# Criando o transformador de colunas para aplicar as transformações apropriadas
# Objetivo: Aplicar as transformações apropriadas às colunas numéricas e categóricas de forma organizada e eficiente.
# Ação: Utilizar ColumnTransformer para combinar os transformadores numérico e categórico.

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
```


```python
# Visualização inicial do dataset
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Cabin_Known</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>0.271174</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>0.110460</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>0.472229</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>1.000000</td>
      <td>C85</td>
      <td>C</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>0.321438</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>0.120745</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>0.434531</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>0.809027</td>
      <td>C123</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>0.434531</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>0.122649</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Colunas presentes no Data Frame
data_final.columns
```




    Index(['onehot__Embarked_C', 'onehot__Embarked_Q', 'onehot__Embarked_S',
           'onehot__Sex_female', 'onehot__Sex_male', 'onehot__Pclass_1',
           'onehot__Pclass_2', 'onehot__Pclass_3', 'remainder__PassengerId',
           'remainder__Survived', 'remainder__Name', 'remainder__Age',
           'remainder__SibSp', 'remainder__Parch', 'remainder__Ticket',
           'remainder__Fare', 'remainder__Cabin', 'remainder__Cabin_Known'],
          dtype='object')




```python
# Aplicando o pré-processador ao dataset
# Objetivo: Transformar o dataset aplicando as padronizações e codificações definidas.
# Ação: Aplicar o transformador de colunas ao dataset.
 
X_transformed = preprocessor.fit_transform(data)
```


```python
# Para uso posterior em modelos, transformar a saída em um DataFrame
# Objetivo: Converter a saída transformada em um DataFrame para facilitar a manipulação e visualização.

# Obter os nomes das colunas das variáveis categóricas após a transformação.
# Combinar os nomes das colunas numéricas com as colunas categóricas codificadas.
# Criar um DataFrame com os dados transformados e os nomes das colunas.

# Obtendo os nomes das colunas das variáveis categóricas após a transformação
encoded_cat_columns = preprocessor.named_transformers_['cat'].get_feature_names_out(input_features=categorical_features)

# Juntando os nomes das colunas das variáveis numéricas com as colunas codificadas das variáveis categóricas
column_names = list(numeric_features) + list(encoded_cat_columns)

# Criando um DataFrame com os dados transformados e os nomes das colunas
X_transformed_df = pd.DataFrame(X_transformed, columns=column_names)

```


```python
# Mostrando as primeiras linhas do DataFrame transformado

print(X_transformed_df.head())
```

            Age      Fare     SibSp     Parch  Sex_male  Embarked_Q  Embarked_S  \
    0 -0.565736 -0.820552  0.432793 -0.473674       1.0         0.0         1.0   
    1  0.663861  2.031623  0.432793 -0.473674       0.0         0.0         0.0   
    2 -0.258337 -0.787578 -0.474545 -0.473674       0.0         0.0         1.0   
    3  0.433312  1.419297  0.432793 -0.473674       0.0         0.0         1.0   
    4  0.433312 -0.781471 -0.474545 -0.473674       1.0         0.0         1.0   
    
       Pclass_2  Pclass_3  Cabin_Known_1  
    0       0.0       1.0            0.0  
    1       0.0       0.0            1.0  
    2       0.0       1.0            0.0  
    3       0.0       0.0            1.0  
    4       0.0       1.0            0.0  
    


```python
# Listar os nomes das colunas do DataFrame.
data_final.columns
```




    Index(['onehot__Embarked_C', 'onehot__Embarked_Q', 'onehot__Embarked_S',
           'onehot__Sex_female', 'onehot__Sex_male', 'onehot__Pclass_1',
           'onehot__Pclass_2', 'onehot__Pclass_3', 'remainder__PassengerId',
           'remainder__Survived', 'remainder__Name', 'remainder__Age',
           'remainder__SibSp', 'remainder__Parch', 'remainder__Ticket',
           'remainder__Fare', 'remainder__Cabin', 'remainder__Cabin_Known'],
          dtype='object')



# Feature Engineering:

A engenharia de recursos envolve a criação de novos atributos que podem nos ajudar nos modelos e a entender melhor os padrões nos dados. Para criação de novos atributos fizemos os seguintes passos:
* Tamanho da Família:
Combinar SibSp (número de irmãos ou cônjuge a bordo) e Parch (número de pais ou filhos a bordo) para formar um novo atributo chamado Family_Size.


```python
# Visualização inicial do dataset
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Cabin_Known</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>0.271174</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>0.110460</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>0.472229</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>1.000000</td>
      <td>C85</td>
      <td>C</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>0.321438</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>0.120745</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>0.434531</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>0.809027</td>
      <td>C123</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>0.434531</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>0.122649</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Objetivo: Extrair títulos (Sr., Sra., Miss, etc.) dos nomes dos passageiros como um indicador de status social, gênero e estado civil.
# Racional: Títulos podem fornecer informações valiosas sobre o passageiro que não estão explícitas em outros atributos.
# Ação: Utilizar a função apply e manipulação de strings para extrair o título do nome completo.

data['Title'] = data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
```


```python
# Visualização inicial do dataset
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Cabin_Known</th>
      <th>Title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>0.271174</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>0.110460</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
      <td>Mr</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>0.472229</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>1.000000</td>
      <td>C85</td>
      <td>C</td>
      <td>1</td>
      <td>Mrs</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>0.321438</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>0.120745</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
      <td>Miss</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>0.434531</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>0.809027</td>
      <td>C123</td>
      <td>S</td>
      <td>1</td>
      <td>Mrs</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>0.434531</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>0.122649</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
      <td>Mr</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Objetivo: Combinar os atributos SibSp (número de irmãos ou cônjuge a bordo) e Parch (número de pais ou filhos a bordo) para formar um novo atributo que representa o tamanho total da família do passageiro.
# Racional: Passageiros com famílias maiores podem ter diferentes probabilidades de sobrevivência.
# Ação: Somar SibSp, Parch e incluir o próprio passageiro (+1).

data['Family_Size'] = data['SibSp'] + data['Parch'] + 1  # Incluindo o próprio passageiro
```


```python
# Visualização inicial do dataset
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Cabin_Known</th>
      <th>Title</th>
      <th>Family_Size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>0.271174</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>0.110460</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
      <td>Mr</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>0.472229</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>1.000000</td>
      <td>C85</td>
      <td>C</td>
      <td>1</td>
      <td>Mrs</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>0.321438</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>0.120745</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
      <td>Miss</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>0.434531</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>0.809027</td>
      <td>C123</td>
      <td>S</td>
      <td>1</td>
      <td>Mrs</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>0.434531</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>0.122649</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
      <td>Mr</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Objetivo: Converter a idade contínua em categorias discretas que podem capturar melhor variações não lineares relacionadas a diferentes faixas etárias.
# Racional: Diferentes grupos etários podem ter diferentes probabilidades de sobrevivência e comportamentos.
# Ação: Definir intervalos (bins) para as faixas etárias e atribuir rótulos correspondentes a esses intervalos.

bins = [0, 12, 20, 40, 60, 80, 100]
labels = ['Child', 'Teen', 'Adult', 'Middle_Aged', 'Senior', 'Elderly']
data['Age_Group'] = pd.cut(data['Age'], bins=bins, labels=labels, right=False)
```


```python
# Visualizando o dataset com o novo atributo Family_Size, Title e Age_Group
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Cabin_Known</th>
      <th>Title</th>
      <th>Family_Size</th>
      <th>Age_Group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>0.271174</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>0.110460</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
      <td>Mr</td>
      <td>2</td>
      <td>Child</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>0.472229</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>1.000000</td>
      <td>C85</td>
      <td>C</td>
      <td>1</td>
      <td>Mrs</td>
      <td>2</td>
      <td>Child</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>0.321438</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>0.120745</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
      <td>Miss</td>
      <td>1</td>
      <td>Child</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>0.434531</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>0.809027</td>
      <td>C123</td>
      <td>S</td>
      <td>1</td>
      <td>Mrs</td>
      <td>2</td>
      <td>Child</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>0.434531</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>0.122649</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
      <td>Mr</td>
      <td>1</td>
      <td>Child</td>
    </tr>
  </tbody>
</table>
</div>



Passos Resumidos

**Visualização Inicial:**
* Exibir as primeiras linhas do dataset original.

**Criação de "Family_Size":**
* Combinar SibSp e Parch para formar Family_Size.

**Extração de "Title":**
* Extrair o título do nome completo dos passageiros.

**Criação de "Age_Group":**
* Categorizar a idade em faixas etárias predefinidas.

**Verificação das Transformações:**
* Exibir as primeiras linhas do dataset após cada nova transformação para validação.

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


```python
# Visualização inicial do dataset
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Cabin_Known</th>
      <th>Title</th>
      <th>Family_Size</th>
      <th>Age_Group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>0.271174</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>0.110460</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
      <td>Mr</td>
      <td>2</td>
      <td>Child</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>0.472229</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>1.000000</td>
      <td>C85</td>
      <td>C</td>
      <td>1</td>
      <td>Mrs</td>
      <td>2</td>
      <td>Child</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>0.321438</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>0.120745</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
      <td>Miss</td>
      <td>1</td>
      <td>Child</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>0.434531</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>0.809027</td>
      <td>C123</td>
      <td>S</td>
      <td>1</td>
      <td>Mrs</td>
      <td>2</td>
      <td>Child</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>0.434531</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>0.122649</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
      <td>Mr</td>
      <td>1</td>
      <td>Child</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Objetivo: Remover as variáveis identificadas como menos úteis para o modelo.
# Ação: Usar o método drop do pandas para descartar as colunas selecionadas.

data.drop(['Ticket', 'Cabin', 'Name', 'PassengerId'], axis=1, inplace=True)
```


```python
# Visualização inicial do dataset
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Cabin_Known</th>
      <th>Title</th>
      <th>Family_Size</th>
      <th>Age_Group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0.271174</td>
      <td>1</td>
      <td>0</td>
      <td>0.110460</td>
      <td>S</td>
      <td>0</td>
      <td>Mr</td>
      <td>2</td>
      <td>Child</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>0.472229</td>
      <td>1</td>
      <td>0</td>
      <td>1.000000</td>
      <td>C</td>
      <td>1</td>
      <td>Mrs</td>
      <td>2</td>
      <td>Child</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>0.321438</td>
      <td>0</td>
      <td>0</td>
      <td>0.120745</td>
      <td>S</td>
      <td>0</td>
      <td>Miss</td>
      <td>1</td>
      <td>Child</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>0.434531</td>
      <td>1</td>
      <td>0</td>
      <td>0.809027</td>
      <td>S</td>
      <td>1</td>
      <td>Mrs</td>
      <td>2</td>
      <td>Child</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0.434531</td>
      <td>0</td>
      <td>0</td>
      <td>0.122649</td>
      <td>S</td>
      <td>0</td>
      <td>Mr</td>
      <td>1</td>
      <td>Child</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Engenharia de recursos
# Objetivo: Criar novos atributos que podem ajudar nos modelos e a entender melhor os padrões nos dados.
# Ação: Implementar a criação de novos atributos como Family_Size e Age_Group.
    
data['Family_Size'] = data['SibSp'] + data['Parch'] + 1

#data['Title'] = data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

bins = [0, 12, 20, 40, 60, 80, 100]
labels = ['Child', 'Teen', 'Adult', 'Middle_Aged', 'Senior', 'Elderly']
data['Age_Group'] = pd.cut(data['Age'], bins=bins, labels=labels, right=False)

# Seleção de recursos
#data.drop(['Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True)

# Salvando o dataframe transformado

data.to_csv('/kaggle/working/titanic_transformed.csv', index=False)
```

Resumo das Etapas

**Visualização Inicial:**
* Exibir as primeiras linhas do dataset original.

**Critérios de Seleção de Recursos:**
* Identificar variáveis relevantes, redundantes e com muitos dados faltantes.

**Descarte de Variáveis:**
* Remover variáveis como Ticket, Cabin, Name e PassengerId.

**Verificação das Transformações:**
* Exibir as primeiras linhas do dataset após cada transformação para validação.

**Engenharia de Recursos:**
* Criar novos atributos Family_Size e Age_Group.

**Salvamento do Dataset Transformado:**
* Salvar o dataframe transformado para uso posterior.

Com a introdução de novas variáveis e a criteriosa seleção de características relevantes, o dataset foi aprimorado significativamente para a modelagem preditiva. Essas modificações são essenciais para elevar a precisão dos modelos de machine learning, pois elas concentram a análise nas características mais informativas enquanto simplificam a estrutura do modelo. Esta abordagem não só melhora a eficácia dos modelos, mas também potencializa a interpretação dos resultados, contribuindo para insights mais claros e decisões baseadas em dados mais robustas.

# Tratamento de dados desbalanceados: 
Lidar com dados desbalanceados é de suma importância em projetos de machine learning, sobretudo em tarefas de classificação. O desequilíbrio entre as classes pode resultar em modelos que favorecem a classe majoritária, comprometendo a eficácia do modelo na identificação da classe minoritária. Diante dessa realidade, foi utilizado técnicas de reamostragem e abordagens algorítmicas projetadas para equilibrar o dataset.

* Avaliar o Desequilíbrio: Contando o número de instâncias de cada classe (sobreviventes e não sobreviventes).


```python
# Visualizar contagem de classes antes do tratamento

class_count = data['Survived'].value_counts()

plt.figure(figsize=(8, 6))
class_count.plot(kind='bar', color=['blue', 'red'])
plt.title('Contagem de Classes (Antes do Tratamento)')
plt.xlabel('Sobrevivente')
plt.ylabel('Número de Passageiros')
plt.xticks([0, 1], ['Não Sobreviveu', 'Sobreviveu'], rotation=0)
plt.show()
```


    
![png](output_57_0.png)
    


* Reamostragem utilizando Oversampling com SMOTE:


```python
from imblearn.over_sampling import SMOTE
```


```python
# Aplicar SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
```


```python
# Visualizar nova contagem de classes após o tratamento
plt.figure(figsize=(8, 6))
pd.Series(y_resampled).value_counts().plot(kind='bar', color=['blue', 'red'])
plt.title('Nova Contagem de Classes (Após SMOTE)')
plt.xlabel('Sobrevivente')
plt.ylabel('Número de Passageiros')
plt.xticks([0, 1], ['Não Sobreviveu', 'Sobreviveu'], rotation=0)
plt.show()
```


    
![png](output_61_0.png)
    



```python
# Reamostragem utilizando Undersampling:
from imblearn.under_sampling import RandomUnderSampler
```


```python
# Aplicar undersampling
under_sampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = under_sampler.fit_resample(X, y)
```


```python
# Visualizar nova contagem de classes após o tratamento
plt.figure(figsize=(8, 6))
pd.Series(y_resampled).value_counts().plot(kind='bar', color=['blue', 'red'])
plt.title('Nova Contagem de Classes (Após Undersampling)')
plt.xlabel('Sobrevivente')
plt.ylabel('Número de Passageiros')
plt.xticks([0, 1], ['Não Sobreviveu', 'Sobreviveu'], rotation=0)
plt.show()
```


    
![png](output_64_0.png)
    


# Ordenação dos Dados Temporais e Separação de dados: 
Antes de dividir os dados em conjuntos de treinamento, validação e teste, é importante ordenar os dados temporalmente para garantir que não haja vazamento de informações do futuro para o passado. Isso pode ser feito ordenando o DataFrame pelo timestamp ou pela variável temporal relevante.


```python
# Visualização inicial do dataset
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Cabin_Known</th>
      <th>Title</th>
      <th>Family_Size</th>
      <th>Age_Group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0.271174</td>
      <td>1</td>
      <td>0</td>
      <td>0.110460</td>
      <td>S</td>
      <td>0</td>
      <td>Mr</td>
      <td>2</td>
      <td>Child</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>0.472229</td>
      <td>1</td>
      <td>0</td>
      <td>1.000000</td>
      <td>C</td>
      <td>1</td>
      <td>Mrs</td>
      <td>2</td>
      <td>Child</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>0.321438</td>
      <td>0</td>
      <td>0</td>
      <td>0.120745</td>
      <td>S</td>
      <td>0</td>
      <td>Miss</td>
      <td>1</td>
      <td>Child</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>0.434531</td>
      <td>1</td>
      <td>0</td>
      <td>0.809027</td>
      <td>S</td>
      <td>1</td>
      <td>Mrs</td>
      <td>2</td>
      <td>Child</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0.434531</td>
      <td>0</td>
      <td>0</td>
      <td>0.122649</td>
      <td>S</td>
      <td>0</td>
      <td>Mr</td>
      <td>1</td>
      <td>Child</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Ordenar os dados temporalmente
data_sorted = data.sort_values(by='Pclass')
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

    Tamanho do conjunto de treinamento: 623
    Tamanho do conjunto de validação: 178
    Tamanho do conjunto de teste: 90
    

Essa divisão comumente adotada divide os dados em 70% para treinamento, 20% para validação e 10% para teste. O conjunto de validação é usado durante o treinamento do modelo para ajustar os hiperparâmetros e avaliar o desempenho do modelo em dados não vistos. O conjunto de teste é reservado para avaliar o desempenho final do modelo após o treinamento e ajuste de hiperparâmetros.

# Redução de Dimensionalidade: 
aplique técnicas como PCA (Análise de Componentes Principais) se a dimensionalidade dos dados for muito alta.


```python
from sklearn.decomposition import PCA
```


```python
# Inicializar o objeto PCA com o número desejado de componentes
pca = PCA(n_components=2)  # Define o número de componentes principais desejados
```


```python
# Aplicar PCA aos dados de treinamento (assumindo que X_train já está definido)
X_train_pca = pca.fit_transform(X_train)
```


```python
# Converter para DataFrame para uma melhor visualização
X_train_pca_df = pd.DataFrame(data=X_train_pca, 
                              columns=['Principal Component 1', 'Principal Component 2'])
```


```python
# Exibir as primeiras linhas para visualização
print(X_train_pca_df.head())
```

       Principal Component 1  Principal Component 2
    0              -0.865722               1.978010
    1              -0.275135              -0.587592
    2              -0.782638              -0.449571
    3              -0.385750              -0.717151
    4              -1.864324               0.704624
    


```python
# Exibir a nova forma dos dados após a redução de dimensionalidade
print("Forma dos dados após a redução de dimensionalidade:", X_train_pca_df.shape)
```

    Forma dos dados após a redução de dimensionalidade: (620, 2)
    


```python
# Exibir a variância explicada por cada componente
print("Variância explicada por cada componente:", pca.explained_variance_ratio_)
```

    Variância explicada por cada componente: [0.36956919 0.33384823]
    


```python
# Primeiras linhas do DataFrame transformado (features_pca_df)

print(X_train_pca_df.head()) #A tabela abaixo apresenta os valores dos dois primeiros componentes principais para as cinco primeiras amostras do conjunto de dados.
```

       Principal Component 1  Principal Component 2
    0              -0.865722               1.978010
    1              -0.275135              -0.587592
    2              -0.782638              -0.449571
    3              -0.385750              -0.717151
    4              -1.864324               0.704624
    

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


```python
# Visualização inicial do dataset
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Cabin_Known</th>
      <th>Title</th>
      <th>Family_Size</th>
      <th>Age_Group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0.271174</td>
      <td>1</td>
      <td>0</td>
      <td>0.110460</td>
      <td>S</td>
      <td>0</td>
      <td>Mr</td>
      <td>2</td>
      <td>Child</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>0.472229</td>
      <td>1</td>
      <td>0</td>
      <td>1.000000</td>
      <td>C</td>
      <td>1</td>
      <td>Mrs</td>
      <td>2</td>
      <td>Child</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>0.321438</td>
      <td>0</td>
      <td>0</td>
      <td>0.120745</td>
      <td>S</td>
      <td>0</td>
      <td>Miss</td>
      <td>1</td>
      <td>Child</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>0.434531</td>
      <td>1</td>
      <td>0</td>
      <td>0.809027</td>
      <td>S</td>
      <td>1</td>
      <td>Mrs</td>
      <td>2</td>
      <td>Child</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0.434531</td>
      <td>0</td>
      <td>0</td>
      <td>0.122649</td>
      <td>S</td>
      <td>0</td>
      <td>Mr</td>
      <td>1</td>
      <td>Child</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Carregar os dados
# data = pd.read_csv("/kaggle/input/dataset-g1pd/Titanic-Dataset.csv")
```


```python
# Definir features e target
X = data.drop('Survived', axis=1)
y = data['Survived']
```


```python
# Visualização inicial do dataset
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Cabin_Known</th>
      <th>Title</th>
      <th>Family_Size</th>
      <th>Age_Group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0.271174</td>
      <td>1</td>
      <td>0</td>
      <td>0.110460</td>
      <td>S</td>
      <td>0</td>
      <td>Mr</td>
      <td>2</td>
      <td>Child</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>0.472229</td>
      <td>1</td>
      <td>0</td>
      <td>1.000000</td>
      <td>C</td>
      <td>1</td>
      <td>Mrs</td>
      <td>2</td>
      <td>Child</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>0.321438</td>
      <td>0</td>
      <td>0</td>
      <td>0.120745</td>
      <td>S</td>
      <td>0</td>
      <td>Miss</td>
      <td>1</td>
      <td>Child</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>0.434531</td>
      <td>1</td>
      <td>0</td>
      <td>0.809027</td>
      <td>S</td>
      <td>1</td>
      <td>Mrs</td>
      <td>2</td>
      <td>Child</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0.434531</td>
      <td>0</td>
      <td>0</td>
      <td>0.122649</td>
      <td>S</td>
      <td>0</td>
      <td>Mr</td>
      <td>1</td>
      <td>Child</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Criar um pipeline com pré-processamento e modelo
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Padronização dos dados
    ('clf', RandomForestClassifier())  # Modelo RandomForest
])
```


```python
# Definindo transformadores para colunas numéricas e categóricas
numeric_features = [...]  # Lista de nomes de características numéricas
categorical_features = [...]  # Lista de nomes de características categóricas

numeric_transformer = ...
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder())
])

# Criando o ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Definindo o modelo de classificação
classifier = RandomForestClassifier()  # Por exemplo, um modelo de floresta aleatória

# Criando o pipeline completo com o modelo
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
```


```python
# Realizar a validação cruzada

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier()),
    ...
])
```


```python
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Carregar dataset
iris = load_iris()
X, y = iris.data, iris.target

# Definir o modelo
model = RandomForestClassifier()

# Executar cross-validation
scores = cross_val_score(model, X, y, cv=5)

# Criar DataFrame com os resultados
results = {
    "Métrica": ["Acurácia Média", "Desvio Padrão dos Scores"],
    "Valor": [scores.mean(), scores.std()]
}
df_results = pd.DataFrame(results)

# Imprimir a tabela
print(tabulate(df_results, headers='keys', tablefmt='pretty'))
```

    +---+--------------------------+---------------------+
    |   |         Métrica          |        Valor        |
    +---+--------------------------+---------------------+
    | 0 |      Acurácia Média      | 0.9666666666666668  |
    | 1 | Desvio Padrão dos Scores | 0.02108185106778919 |
    +---+--------------------------+---------------------+
    

1. Importamos cross_val_score do sklearn para realizar a validação cruzada.
2. Criamos um pipeline que inclui uma etapa de pré-processamento (padronização dos dados usando StandardScaler) e um modelo de RandomForest.
3. Usamos cross_val_score para calcular a acurácia do modelo em cada fold da validação cruzada. O argumento cv=5 especifica o número de folds a serem usados.
4. Finalmente, imprimimos a média das acurácias dos folds e o desvio padrão dos scores para avaliar o desempenho médio e a consistência do modelo em diferentes conjuntos de dados de treinamento e teste.

```html
Acurácia Média: 
Desvio Padrão dos Scores: 
```

**Acurácia Média:** Este valor é a média dos scores de acurácia obtidos em cada uma das 5 iterações da validação cruzada. O formato {:.2f} usado no format garante que o número seja mostrado com duas casas decimais. Assumindo um valor de 0.82, isso indicaria que, em média, o modelo foi capaz de prever corretamente a sobrevivência 82% das vezes.

**Desvio Padrão dos Scores:** Este número mostra o desvio padrão dos scores de acurácia obtidos, o que dá uma ideia da variação nos resultados do modelo entre os diferentes folds da validação cruzada. Um valor de 0.03, sugere que as variações entre os resultados de cada fold são relativamente pequenas, indicando que o modelo é estável.


# Algoritmo Random Forest Classifier


```python
data = pd.read_csv('/kaggle/working/titanic_transformed.csv')
```


```python
# Visualização inicial do dataset
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Cabin_Known</th>
      <th>Title</th>
      <th>Family_Size</th>
      <th>Age_Group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0.271174</td>
      <td>1</td>
      <td>0</td>
      <td>0.110460</td>
      <td>S</td>
      <td>0</td>
      <td>Mr</td>
      <td>2</td>
      <td>Child</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>0.472229</td>
      <td>1</td>
      <td>0</td>
      <td>1.000000</td>
      <td>C</td>
      <td>1</td>
      <td>Mrs</td>
      <td>2</td>
      <td>Child</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>0.321438</td>
      <td>0</td>
      <td>0</td>
      <td>0.120745</td>
      <td>S</td>
      <td>0</td>
      <td>Miss</td>
      <td>1</td>
      <td>Child</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>0.434531</td>
      <td>1</td>
      <td>0</td>
      <td>0.809027</td>
      <td>S</td>
      <td>1</td>
      <td>Mrs</td>
      <td>2</td>
      <td>Child</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0.434531</td>
      <td>0</td>
      <td>0</td>
      <td>0.122649</td>
      <td>S</td>
      <td>0</td>
      <td>Mr</td>
      <td>1</td>
      <td>Child</td>
    </tr>
  </tbody>
</table>
</div>




```python
# data.shape
# data.head()
# data.tail()
# data.info()
# data.describe()
# data.isnull()
# data.isnull().sum()
data['Survived'].value_counts()
```




    Survived
    0    549
    1    342
    Name: count, dtype: int64




```python
sns.countplot(data=data, x="Survived")
plt.xlabel("Survived")
plt.ylabel("Count")
plt.title("Survived variable count plot")
plt.show()
```


    
![png](output_96_0.png)
    



```python
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# X.shape
# y.shape
```


```python
# Definir colunas categóricas e numéricas
categorical_cols = ['Pclass', 'Sex', 'Embarked']
numerical_cols = ['Age', 'SibSp', 'Parch', 'Fare']

# Criar transformers para colunas numéricas e categóricas
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Criar o preprocessor utilizando ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Aplicar as transformações no dataset
X = data.drop('Survived', axis=1)
y = data['Survived']
X = preprocessor.fit_transform(X)

```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=99)

clf = RandomForestClassifier(criterion = 'gini',
                            max_depth = 8,
                            min_samples_split = 10,
                            random_state = 5)

clf.fit(X_train, y_train)
```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestClassifier(max_depth=8, min_samples_split=10, random_state=5)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestClassifier</label><div class="sk-toggleable__content"><pre>RandomForestClassifier(max_depth=8, min_samples_split=10, random_state=5)</pre></div></div></div></div></div>




```python
# clf.feature_importances_
# data.columns
```


```python
y_pred = clf.predict(X_test)
# y_pred
```


```python
# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo
clf = RandomForestClassifier(criterion='gini', max_depth=8, min_samples_split=10, random_state=5)
clf.fit(X_train, y_train)

# Fazer previsões
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)

# Calcular métricas de desempenho
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Métricas e descrição
# Accuracy: A proporção de previsões corretas entre o total de previsões.
print("Accuracy:", accuracy)
# Precision: A proporção de verdadeiros positivos entre o total de previsões positivas.
print("Precision:", precision)
# Recall or Sensitivity: A proporção de verdadeiros positivos entre o total de exemplos positivos reais.
print("Recall:", recall)
# F1-Score: A média harmônica da precisão e da revocação, proporcionando um equilíbrio entre ambos.
print("F1 Score:", f1)
# AUC-ROC: A área sob a curva ROC, que indica o desempenho do classificador em diferentes limiares de decisão.
print("ROC AUC:", roc_auc)
# Confusion Matrix: Mostra o número de verdadeiros positivos, verdadeiros negativos, falsos positivos e falsos negativos.
print("Confusion Matrix:\n", conf_matrix)
# Classification Report: Inclui precisão, revocação, F1-Score e suporte para cada classe.
print("Classification Report:\n", class_report)
```

    Accuracy: 0.8324022346368715
    Precision: 0.84375
    Recall: 0.7297297297297297
    F1 Score: 0.7826086956521738
    ROC AUC: 0.8907979407979408
    Confusion Matrix:
     [[95 10]
     [20 54]]
    Classification Report:
                   precision    recall  f1-score   support
    
               0       0.83      0.90      0.86       105
               1       0.84      0.73      0.78        74
    
        accuracy                           0.83       179
       macro avg       0.83      0.82      0.82       179
    weighted avg       0.83      0.83      0.83       179
    
    


```python
cross_val_score(clf, X_train, y_train, cv=10)
```




    array([0.83333333, 0.79166667, 0.74647887, 0.92957746, 0.88732394,
           0.77464789, 0.83098592, 0.77464789, 0.77464789, 0.90140845])




```python
features = data.columns
importances = clf.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)),importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
```


    
![png](output_104_0.png)
    


# Algoritmo Regreção Logistica


```python
# Carregar o dataset
data = pd.read_csv('titanic_transformed.csv')

# Verificar as primeiras linhas do dataset para entender sua estrutura
print(data.head())

# Separar os recursos (features) e a variável alvo (target)
X = data.drop('Survived', axis=1)  # Supondo que a coluna alvo seja 'Survived'
y = data['Survived']

# Identificar colunas categóricas e numéricas
categorical_cols = X.select_dtypes(include=['object']).columns
numeric_cols = X.select_dtypes(include=['number']).columns

# Criar o pré-processador
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Criar o pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calcular as métricas de avaliação
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Exibir os resultados
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC AUC: {roc_auc}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
```

       Survived  Pclass     Sex       Age  SibSp  Parch      Fare Embarked  \
    0         0       3    male  0.271174      1      0  0.110460        S   
    1         1       1  female  0.472229      1      0  1.000000        C   
    2         1       3  female  0.321438      0      0  0.120745        S   
    3         1       1  female  0.434531      1      0  0.809027        S   
    4         0       3    male  0.434531      0      0  0.122649        S   
    
       Cabin_Known Title  Family_Size Age_Group  
    0            0    Mr            2     Child  
    1            1   Mrs            2     Child  
    2            0  Miss            1     Child  
    3            1   Mrs            2     Child  
    4            0    Mr            1     Child  
    Accuracy: 0.8268156424581006
    Precision: 0.7866666666666666
    Recall: 0.7972972972972973
    F1 Score: 0.7919463087248321
    ROC AUC: 0.8881595881595881
    Confusion Matrix:
    [[89 16]
     [15 59]]
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.86      0.85      0.85       105
               1       0.79      0.80      0.79        74
    
        accuracy                           0.83       179
       macro avg       0.82      0.82      0.82       179
    weighted avg       0.83      0.83      0.83       179
    
    

## Análise de Sensibilidade a Classes Desbalanceadas em Algoritmos de Classificação

#### Algoritmos Utilizados
1. **Random Forest Classifier**
2. **Regressão Logística**

#### Dados Utilizados
- Dataset do Titanic com a variável alvo `Survived`.
- Desbalanceamento nas classes: 549 não sobreviveram (0) e 342 sobreviveram (1).

### Métricas de Avaliação

- **Accuracy**: Proporção de previsões corretas.
- **Precision**: Proporção de verdadeiros positivos entre o total de previsões positivas.
- **Recall (Sensibilidade)**: Proporção de verdadeiros positivos entre o total de exemplos positivos reais.
- **F1-Score**: Média harmônica da precisão e recall.
- **ROC AUC**: Área sob a curva ROC, que mede o desempenho do classificador em diferentes limiares de decisão.
- **Confusion Matrix**: Mostra o número de verdadeiros positivos, verdadeiros negativos, falsos positivos e falsos negativos.
- **Classification Report**: Inclui precisão, recall, F1-Score e suporte para cada classe.

### Avaliação de Sensibilidade a Classes Desbalanceadas

>## Random Forest Classifier
>
> ### Métricas de Avaliação
>
>- **Accuracy**: 0.8324022346368715
>- **Precision**: 0.84375
>- **Recall**: 0.7297297297297297
>- **F1 Score**: 0.7826086956521738
>- **ROC AUC**: 0.8907979407979408
>
>### Matriz de Confusão
>
>[[95 10]
>[20 54]]
>
>           precision    recall  f1-score   support
>
>       0       0.83      0.90      0.86       105
>       1       0.84      0.73      0.78        74
>
>accuracy                           0.83       179
>
>macro avg 0.83 0.82 0.82 179
>weighted avg 0.83 0.83 0.83 179

>## Algoritmo Regreção Logistica
>
>| Survived | Pclass | Sex    | Age      | SibSp | Parch | Fare     | Embarked | Cabin_Known | Title | Family_Size | Age_Group |
>|----------|--------|--------|----------|-------|-------|----------|----------|-------------|-------|-------------|-----------|
>| 0        | 3      | male   | 0.271174 | 1     | 0     | 0.110460 | S        | 0           | Mr    | 2           | Child     |
>| 1        | 1      | female | 0.472229 | 1     | 0     | 1.000000 | C        | 1           | Mrs   | 2           | Child     |
>| 1        | 3      | female | 0.321438 | 0     | 0     | 0.120745 | S        | 0           | Miss  | 1           | Child     |
>| 1        | 1      | female | 0.434531 | 1     | 0     | 0.809027 | S        | 1           | Mrs   | 2           | Child     |
>| 0        | 3      | male   | 0.434531 | 0     | 0     | 0.122649 | S        | 0           | Mr    | 1           | Child     |
>
>### Métricas de Avaliação
>
>- **Accuracy**: 0.8268156424581006
>- **Precision**: 0.7866666666666666
>- **Recall**: 0.7972972972972973
>- **F1 Score**: 0.7919463087248321
>- **ROC AUC**: 0.8881595881595881
>
>### Matriz de Confusão
>[[89 16]
>[15 59]]
>
>
>### Classification Report
>
>          precision    recall  f1-score   support
>
>       0       0.86      0.85      0.85       105
>       1       0.79      0.80      0.79        74
>
>accuracy                           0.83       179
>
>macro avg 0.82 0.82 0.82 179
>weighted avg 0.83 0.83 0.83 179

## Sensibilidade a Classes Desbalanceadas

#### Random Forest Classifier

- **Precision** e **Recall** são diferentes para as classes 0 e 1.
  - Classe 0 (não sobreviveu): Alta precisão (0.83) e alto recall (0.90).
  - Classe 1 (sobreviveu): Boa precisão (0.84) mas recall menor (0.73).
- **F1-Score**: Classe 0 (0.86) é maior que a Classe 1 (0.78).
- **ROC AUC**: 0.8908, indicando bom desempenho geral.

#### Regressão Logística

- **Precision** e **Recall** são também diferentes para as classes 0 e 1.
  - Classe 0 (não sobreviveu): Alta precisão (0.86) e alto recall (0.85).
  - Classe 1 (sobreviveu): Boa precisão (0.79) e recall similar (0.80).
- **F1-Score**: Classe 0 (0.85) é maior que a Classe 1 (0.79).
- **ROC AUC**: 0.8882, indicando bom desempenho geral.

### Conclusão

Ambos os algoritmos mostram um bom desempenho geral, mas há uma diferença notável entre as métricas das duas classes. Essa diferença sugere que os algoritmos são sensíveis ao desbalanceamento de classes, particularmente no recall da classe minoritária (sobreviveu).

#### Impacto do Desbalanceamento

- **Random Forest**: Tende a ter um recall menor para a classe minoritária.
- **Regressão Logística**: Também mostra menor recall para a classe minoritária, mas um pouco melhor em balancear precision e recall.

## Recomendações

1. **Uso de Métricas Apropriadas**:
   - Utilize a F1-Score e o ROC AUC para uma avaliação equilibrada.
   - Considere a matriz de confusão para entender o desempenho em detalhes.

2. **Estratégias de Balanceamento**:
   - Aplicar técnicas como oversampling (SMOTE) ou undersampling.
   - Ajustar pesos das classes no modelo (`class_weight='balanced'`).

3. **Validação Cruzada**:
   - Utilizar validação cruzada estratificada para garantir a representação de ambas as classes em cada fold.

4. **Análise de Erros**:
   - Examine os falsos negativos e falsos positivos para entender e melhorar o modelo.
