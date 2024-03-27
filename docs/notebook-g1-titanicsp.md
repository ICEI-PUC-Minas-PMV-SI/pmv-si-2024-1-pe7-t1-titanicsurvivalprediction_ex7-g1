**1 - Importação das Bibliotecas/Dados**


```python
# Grupo 1 - notebook_titanicsurvivalprediction

import numpy as np # Manipulação de matrizes
import pandas as pd # Criação e manipulação de dataset
%matplotlib inline
from pandas import *
import matplotlib.pyplot as plt # Plotagem de dados
import matplotlib.font_manager
import seaborn as sns # Plotagem e visualização dos dados
from tabulate import tabulate
from scipy.stats import chi2_contingency

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```

    /kaggle/input/titanic-dataset/Titanic-Dataset.csv
    


```python
# Criando dataset
df = pd.read_csv("/kaggle/input/titanic-dataset/Titanic-Dataset.csv")
```

**2 - Explorando os Dados**


```python
# Visualização inicial do dataset
df.head()
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
# Número de linhas do Data Frame
df.shape
```




    (891, 12)




```python
# Colunas presentes no Data Frame
df.columns
```




    Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
           'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
          dtype='object')




```python
# Informações sobre o conjunto de dados
print("\nInformações sobre o conjunto de dados:")
print(df.info())
```

    
    Informações sobre o conjunto de dados:
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  891 non-null    int64  
     1   Survived     891 non-null    int64  
     2   Pclass       891 non-null    int64  
     3   Name         891 non-null    object 
     4   Sex          891 non-null    object 
     5   Age          714 non-null    float64
     6   SibSp        891 non-null    int64  
     7   Parch        891 non-null    int64  
     8   Ticket       891 non-null    object 
     9   Fare         891 non-null    float64
     10  Cabin        204 non-null    object 
     11  Embarked     889 non-null    object 
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.7+ KB
    None
    


```python
# Avaliando a existência de dados nulos
df.isnull().sum()
```




    PassengerId      0
    Survived         0
    Pclass           0
    Name             0
    Sex              0
    Age            177
    SibSp            0
    Parch            0
    Ticket           0
    Fare             0
    Cabin          687
    Embarked         2
    dtype: int64




```python
# Avaliando valores nulos
df[df.isnull().any(axis=1)]
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
    <tr>
      <th>5</th>
      <td>6</td>
      <td>0</td>
      <td>3</td>
      <td>Moran, Mr. James</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>330877</td>
      <td>8.4583</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>0</td>
      <td>3</td>
      <td>Palsson, Master. Gosta Leonard</td>
      <td>male</td>
      <td>2.0</td>
      <td>3</td>
      <td>1</td>
      <td>349909</td>
      <td>21.0750</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>884</th>
      <td>885</td>
      <td>0</td>
      <td>3</td>
      <td>Sutehall, Mr. Henry Jr</td>
      <td>male</td>
      <td>25.0</td>
      <td>0</td>
      <td>0</td>
      <td>SOTON/OQ 392076</td>
      <td>7.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>885</th>
      <td>886</td>
      <td>0</td>
      <td>3</td>
      <td>Rice, Mrs. William (Margaret Norton)</td>
      <td>female</td>
      <td>39.0</td>
      <td>0</td>
      <td>5</td>
      <td>382652</td>
      <td>29.1250</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>886</th>
      <td>887</td>
      <td>0</td>
      <td>2</td>
      <td>Montvila, Rev. Juozas</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>211536</td>
      <td>13.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>888</th>
      <td>889</td>
      <td>0</td>
      <td>3</td>
      <td>Johnston, Miss. Catherine Helen "Carrie"</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>W./C. 6607</td>
      <td>23.4500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>890</th>
      <td>891</td>
      <td>0</td>
      <td>3</td>
      <td>Dooley, Mr. Patrick</td>
      <td>male</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>370376</td>
      <td>7.7500</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
  </tbody>
</table>
<p>708 rows × 12 columns</p>
</div>




```python
# Calculando a porcentagem de valores não nulos em cada coluna
total_rows = len(df)
percentages = (df.count() / total_rows) * 100

# Mostrando os valores em porcentagem
print("\nPorcentagem de valores não nulos em cada coluna:")
print(percentages)
```

    
    Porcentagem de valores não nulos em cada coluna:
    PassengerId    100.000000
    Survived       100.000000
    Pclass         100.000000
    Name           100.000000
    Sex            100.000000
    Age             80.134680
    SibSp          100.000000
    Parch          100.000000
    Ticket         100.000000
    Fare           100.000000
    Cabin           22.895623
    Embarked        99.775533
    dtype: float64
    


```python
# Remover avisos
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)

# Gráfico: Distribuição da sobrevivência por sexo
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Survived', hue='Sex')
plt.title('Distribuição da Sobrevivência por Sexo')
plt.xlabel('Sobreviveu?')
plt.ylabel('Contagem')
plt.show()

# Gráfico: Distribuição da sobrevivência por classe
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Survived', hue='Pclass')
plt.title('Distribuição da Sobrevivência por Classe')
plt.xlabel('Sobreviveu?')
plt.ylabel('Contagem')
plt.show()

# Gráfico: Distribuição da sobrevivência por dependentes
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Survived', hue='SibSp')
plt.title('Distribuição da Sobrevivência por Dependentes-Irmãos/Cônjuge')
plt.xlabel('Sobreviveu?')
plt.ylabel('Contagem')
plt.show()

# Gráfico: Distribuição da sobrevivência por dependentes
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Survived', hue='Parch')
plt.title('Distribuição da Sobrevivência por Dependentes-Pais/Filhos')
plt.xlabel('Sobreviveu?')
plt.ylabel('Contagem')
plt.show()

# Gráfico: Distribuição da sobrevivência por idade
plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='Age', hue='Survived', kde=True)
plt.title('Distribuição da Sobrevivência por Idade')
plt.xlabel('Idade')
plt.ylabel('Contagem')
plt.show()

```


    
![png](output_11_0.png)
    



    
![png](output_11_1.png)
    



    
![png](output_11_2.png)
    



    
![png](output_11_3.png)
    


    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1075: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1075: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    


    
![png](output_11_5.png)
    



```python
# Calcula as estatísticas descritivas
statistics = df.describe(include='all')

# Imprimir as estatísticas descritivas em uma tabela formatada
print("Estatísticas Descritivas:")
print(tabulate(statistics, headers='keys', tablefmt='fancy_grid'))

# Ajustando largura das colunas
colalign = ['center'] * len(statistics.columns)

# Converte os dados em uma tabela formatada
table = tabulate(statistics, headers='keys', tablefmt='pretty', showindex=True, colalign=colalign)
```

    Estatísticas Descritivas:
    ╒════════╤═══════════════╤════════════╤════════════╤═════════════════════════╤═══════╤══════════╤════════════╤════════════╤══════════╤══════════╤═════════╤════════════╕
    │        │   PassengerId │   Survived │     Pclass │ Name                    │ Sex   │      Age │      SibSp │      Parch │   Ticket │     Fare │ Cabin   │ Embarked   │
    ╞════════╪═══════════════╪════════════╪════════════╪═════════════════════════╪═══════╪══════════╪════════════╪════════════╪══════════╪══════════╪═════════╪════════════╡
    │ count  │       891     │ 891        │ 891        │ 891                     │ 891   │ 714      │ 891        │ 891        │      891 │ 891      │ 204     │ 889        │
    ├────────┼───────────────┼────────────┼────────────┼─────────────────────────┼───────┼──────────┼────────────┼────────────┼──────────┼──────────┼─────────┼────────────┤
    │ unique │       nan     │ nan        │ nan        │ 891                     │ 2     │ nan      │ nan        │ nan        │      681 │ nan      │ 147     │ 3          │
    ├────────┼───────────────┼────────────┼────────────┼─────────────────────────┼───────┼──────────┼────────────┼────────────┼──────────┼──────────┼─────────┼────────────┤
    │ top    │       nan     │ nan        │ nan        │ Braund, Mr. Owen Harris │ male  │ nan      │ nan        │ nan        │   347082 │ nan      │ B96 B98 │ S          │
    ├────────┼───────────────┼────────────┼────────────┼─────────────────────────┼───────┼──────────┼────────────┼────────────┼──────────┼──────────┼─────────┼────────────┤
    │ freq   │       nan     │ nan        │ nan        │ 1                       │ 577   │ nan      │ nan        │ nan        │        7 │ nan      │ 4       │ 644        │
    ├────────┼───────────────┼────────────┼────────────┼─────────────────────────┼───────┼──────────┼────────────┼────────────┼──────────┼──────────┼─────────┼────────────┤
    │ mean   │       446     │   0.383838 │   2.30864  │ nan                     │ nan   │  29.6991 │   0.523008 │   0.381594 │      nan │  32.2042 │ nan     │ nan        │
    ├────────┼───────────────┼────────────┼────────────┼─────────────────────────┼───────┼──────────┼────────────┼────────────┼──────────┼──────────┼─────────┼────────────┤
    │ std    │       257.354 │   0.486592 │   0.836071 │ nan                     │ nan   │  14.5265 │   1.10274  │   0.806057 │      nan │  49.6934 │ nan     │ nan        │
    ├────────┼───────────────┼────────────┼────────────┼─────────────────────────┼───────┼──────────┼────────────┼────────────┼──────────┼──────────┼─────────┼────────────┤
    │ min    │         1     │   0        │   1        │ nan                     │ nan   │   0.42   │   0        │   0        │      nan │   0      │ nan     │ nan        │
    ├────────┼───────────────┼────────────┼────────────┼─────────────────────────┼───────┼──────────┼────────────┼────────────┼──────────┼──────────┼─────────┼────────────┤
    │ 25%    │       223.5   │   0        │   2        │ nan                     │ nan   │  20.125  │   0        │   0        │      nan │   7.9104 │ nan     │ nan        │
    ├────────┼───────────────┼────────────┼────────────┼─────────────────────────┼───────┼──────────┼────────────┼────────────┼──────────┼──────────┼─────────┼────────────┤
    │ 50%    │       446     │   0        │   3        │ nan                     │ nan   │  28      │   0        │   0        │      nan │  14.4542 │ nan     │ nan        │
    ├────────┼───────────────┼────────────┼────────────┼─────────────────────────┼───────┼──────────┼────────────┼────────────┼──────────┼──────────┼─────────┼────────────┤
    │ 75%    │       668.5   │   1        │   3        │ nan                     │ nan   │  38      │   1        │   0        │      nan │  31      │ nan     │ nan        │
    ├────────┼───────────────┼────────────┼────────────┼─────────────────────────┼───────┼──────────┼────────────┼────────────┼──────────┼──────────┼─────────┼────────────┤
    │ max    │       891     │   1        │   3        │ nan                     │ nan   │  80      │   8        │   6        │      nan │ 512.329  │ nan     │ nan        │
    ╘════════╧═══════════════╧════════════╧════════════╧═════════════════════════╧═══════╧══════════╧════════════╧════════════╧══════════╧══════════╧═════════╧════════════╛
    


```python
# Calcular as taxas de sobrevivência
survival_rate_sex = df.groupby('Sex')['Survived'].mean()
survival_rate_class = df.groupby('Pclass')['Survived'].mean()
survival_rate_age = df.groupby(pd.cut(df['Age'], bins=[0, 18, 30, 50, 100]))['Survived'].mean()
survival_rate_SibSp = df.groupby('SibSp')['Survived'].mean()
survival_rate_Parch = df.groupby('SibSp')['Survived'].mean()

# Memória de Cálculo
print("Memória de Cálculo:")
print("Taxa de Sobrevivência por Sexo:")
print(tabulate(survival_rate_sex.reset_index(), headers='keys', tablefmt='fancy_grid'))
print("\nTaxa de Sobrevivência por Classe Socioeconômica:")
print(tabulate(survival_rate_class.reset_index(), headers='keys', tablefmt='fancy_grid'))
print("\nTaxa de Sobrevivência por Faixa Etária:")
print(tabulate(survival_rate_age.reset_index(), headers='keys', tablefmt='fancy_grid'))
print("\nTaxa de Sobrevivência por Faixa Dependentes-Irmãos/Cônjuge:")
print(tabulate(survival_rate_SibSp.reset_index(), headers='keys', tablefmt='fancy_grid'))
print("\nTaxa de Sobrevivência por Faixa Dependentes-Pais/Filhos:")
print(tabulate(survival_rate_Parch.reset_index(), headers='keys', tablefmt='fancy_grid'))

# Fórmulas
print("\nFórmulas Utilizadas:")
print("Taxa de Sobrevivência por Sexo: (Número de Sobreviventes do Sexo / Número Total de Passageiros do Sexo)")
print("Taxa de Sobrevivência por Classe Socioeconômica: (Número de Sobreviventes da Classe / Número Total de Passageiros da Classe)")
print("Taxa de Sobrevivência por Faixa Etária: (Número de Sobreviventes na Faixa Etária / Número Total de Passageiros na Faixa Etária)")
print("Taxa de Sobrevivência por Faixa Etária: (Número de Sobreviventes de Dependentes-Irmãos/Cônjuge / Número Total de Passageiros Dependentes-Irmãos/Cônjuge)")
print("Taxa de Sobrevivência por Faixa Etária: (Número de Sobreviventes de Dependentes-Pais/Filhos / Número Total de Passageiros Dependentes-Pais/Filhos)")

```

    Memória de Cálculo:
    Taxa de Sobrevivência por Sexo:
    ╒════╤════════╤════════════╕
    │    │ Sex    │   Survived │
    ╞════╪════════╪════════════╡
    │  0 │ female │   0.742038 │
    ├────┼────────┼────────────┤
    │  1 │ male   │   0.188908 │
    ╘════╧════════╧════════════╛
    
    Taxa de Sobrevivência por Classe Socioeconômica:
    ╒════╤══════════╤════════════╕
    │    │   Pclass │   Survived │
    ╞════╪══════════╪════════════╡
    │  0 │        1 │   0.62963  │
    ├────┼──────────┼────────────┤
    │  1 │        2 │   0.472826 │
    ├────┼──────────┼────────────┤
    │  2 │        3 │   0.242363 │
    ╘════╧══════════╧════════════╛
    
    Taxa de Sobrevivência por Faixa Etária:
    ╒════╤═══════════╤════════════╕
    │    │ Age       │   Survived │
    ╞════╪═══════════╪════════════╡
    │  0 │ (0, 18]   │   0.503597 │
    ├────┼───────────┼────────────┤
    │  1 │ (18, 30]  │   0.355556 │
    ├────┼───────────┼────────────┤
    │  2 │ (30, 50]  │   0.423237 │
    ├────┼───────────┼────────────┤
    │  3 │ (50, 100] │   0.34375  │
    ╘════╧═══════════╧════════════╛
    
    Taxa de Sobrevivência por Faixa Dependentes-Irmãos/Cônjuge:
    ╒════╤═════════╤════════════╕
    │    │   SibSp │   Survived │
    ╞════╪═════════╪════════════╡
    │  0 │       0 │   0.345395 │
    ├────┼─────────┼────────────┤
    │  1 │       1 │   0.535885 │
    ├────┼─────────┼────────────┤
    │  2 │       2 │   0.464286 │
    ├────┼─────────┼────────────┤
    │  3 │       3 │   0.25     │
    ├────┼─────────┼────────────┤
    │  4 │       4 │   0.166667 │
    ├────┼─────────┼────────────┤
    │  5 │       5 │   0        │
    ├────┼─────────┼────────────┤
    │  6 │       8 │   0        │
    ╘════╧═════════╧════════════╛
    
    Taxa de Sobrevivência por Faixa Dependentes-Pais/Filhos:
    ╒════╤═════════╤════════════╕
    │    │   SibSp │   Survived │
    ╞════╪═════════╪════════════╡
    │  0 │       0 │   0.345395 │
    ├────┼─────────┼────────────┤
    │  1 │       1 │   0.535885 │
    ├────┼─────────┼────────────┤
    │  2 │       2 │   0.464286 │
    ├────┼─────────┼────────────┤
    │  3 │       3 │   0.25     │
    ├────┼─────────┼────────────┤
    │  4 │       4 │   0.166667 │
    ├────┼─────────┼────────────┤
    │  5 │       5 │   0        │
    ├────┼─────────┼────────────┤
    │  6 │       8 │   0        │
    ╘════╧═════════╧════════════╛
    
    Fórmulas Utilizadas:
    Taxa de Sobrevivência por Sexo: (Número de Sobreviventes do Sexo / Número Total de Passageiros do Sexo)
    Taxa de Sobrevivência por Classe Socioeconômica: (Número de Sobreviventes da Classe / Número Total de Passageiros da Classe)
    Taxa de Sobrevivência por Faixa Etária: (Número de Sobreviventes na Faixa Etária / Número Total de Passageiros na Faixa Etária)
    Taxa de Sobrevivência por Faixa Etária: (Número de Sobreviventes de Dependentes-Irmãos/Cônjuge / Número Total de Passageiros Dependentes-Irmãos/Cônjuge)
    Taxa de Sobrevivência por Faixa Etária: (Número de Sobreviventes de Dependentes-Pais/Filhos / Número Total de Passageiros Dependentes-Pais/Filhos)
    

    /tmp/ipykernel_17/843437112.py:4: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      survival_rate_age = df.groupby(pd.cut(df['Age'], bins=[0, 18, 30, 50, 100]))['Survived'].mean()
    


```python
# matriz de correlação
plt.figure(figsize=(30,15))
sns.pairplot(df, x_vars=['Parch', 'SibSp', 'Age', 'Pclass'], y_vars=['Parch', 'SibSp', 'Age', 'Pclass'])# Calcula a matriz de correlação
```

    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    




    <seaborn.axisgrid.PairGrid at 0x78b6fadafc10>




    <Figure size 3000x1500 with 0 Axes>



    
![png](output_14_3.png)
    



```python
plt.figure(figsize=(15,15))
sns.heatmap(df.corr(), annot=True, cmap='Oranges', fmt='.2f')
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In[14], line 2
          1 plt.figure(figsize=(15,15))
    ----> 2 sns.heatmap(df.corr(), annot=True, cmap='Oranges', fmt='.2f')
    

    File /opt/conda/lib/python3.10/site-packages/pandas/core/frame.py:11036, in DataFrame.corr(self, method, min_periods, numeric_only)
      11034 cols = data.columns
      11035 idx = cols.copy()
    > 11036 mat = data.to_numpy(dtype=float, na_value=np.nan, copy=False)
      11038 if method == "pearson":
      11039     correl = libalgos.nancorr(mat, minp=min_periods)
    

    File /opt/conda/lib/python3.10/site-packages/pandas/core/frame.py:1981, in DataFrame.to_numpy(self, dtype, copy, na_value)
       1979 if dtype is not None:
       1980     dtype = np.dtype(dtype)
    -> 1981 result = self._mgr.as_array(dtype=dtype, copy=copy, na_value=na_value)
       1982 if result.dtype is not dtype:
       1983     result = np.array(result, dtype=dtype, copy=False)
    

    File /opt/conda/lib/python3.10/site-packages/pandas/core/internals/managers.py:1692, in BlockManager.as_array(self, dtype, copy, na_value)
       1690         arr.flags.writeable = False
       1691 else:
    -> 1692     arr = self._interleave(dtype=dtype, na_value=na_value)
       1693     # The underlying data was copied within _interleave, so no need
       1694     # to further copy if copy=True or setting na_value
       1696 if na_value is lib.no_default:
    

    File /opt/conda/lib/python3.10/site-packages/pandas/core/internals/managers.py:1751, in BlockManager._interleave(self, dtype, na_value)
       1749     else:
       1750         arr = blk.get_values(dtype)
    -> 1751     result[rl.indexer] = arr
       1752     itemmask[rl.indexer] = 1
       1754 if not itemmask.all():
    

    ValueError: could not convert string to float: 'Braund, Mr. Owen Harris'



    <Figure size 1500x1500 with 0 Axes>

