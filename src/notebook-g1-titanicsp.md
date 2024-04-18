Projeto: Pesquisa e Experimentação em Sistemas de Informação-Grupo 1
notebook_titanicsurvivalprediction


```python
# Grupo 1 - notebook_titanicsurvivalprediction

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

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```


```python
# Criando dataset
df = pd.read_csv("/kaggle/input/titanic-dataset/Titanic-Dataset.csv")
```


```python
# Visualização inicial do dataset
df.head()
```




<div>

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
#histograma das variáveis numéricas
df.hist(figsize=(13,9))
```




    array([[<Axes: title={'center': 'PassengerId'}>,
            <Axes: title={'center': 'Survived'}>,
            <Axes: title={'center': 'Pclass'}>],
           [<Axes: title={'center': 'Age'}>,
            <Axes: title={'center': 'SibSp'}>,
            <Axes: title={'center': 'Parch'}>],
           [<Axes: title={'center': 'Fare'}>, <Axes: >, <Axes: >]],
          dtype=object)




    
![](/docs/img/output_10_1.png)
    



```python
#definindo os dados que aparecerão no gráfico

labels = ['Não Sobreviventes', "Sobreviventes"] 

contagem = pd.cut(x=df.Survived, bins=2,labels= labels, include_lowest=True).value_counts() #nº de sobreviventes

taxa_de_sobreviventes = (pd.value_counts(pd.cut(x=df.Survived, bins=2,labels= labels, 
                                                include_lowest=True),normalize=True) * 100).round(1) #taxa de Sobreviventes
quant_sobrevi = pd.DataFrame({"Contagem":contagem, 
                              'Taxa de Sobrevivência(%)':taxa_de_sobreviventes}) #criando um DataFrame para facilitar a visualização dos dados
                              
quant_sobrevi
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Contagem</th>
      <th>Taxa de Sobrevivência(%)</th>
    </tr>
    <tr>
      <th>Survived</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Não Sobreviventes</th>
      <td>549</td>
      <td>61.6</td>
    </tr>
    <tr>
      <th>Sobreviventes</th>
      <td>342</td>
      <td>38.4</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(6, 4))

cores= ['#009ACD', '#ADD8E6']
percentages = list(quant_sobrevi['Taxa de Sobrevivência(%)'])
explode=(0.1,0)

plt.pie(percentages, explode=explode, 
       labels=labels,
       colors = cores,
       autopct='%1.0f%%',
       shadow=True, startangle=0,   
       pctdistance=0.5,labeldistance=1.1)
plt.title("Taxa de Sobreviventes do Titanic", fontsize=20, pad=20)
```




    Text(0.5, 1.0, 'Taxa de Sobreviventes do Titanic')




    
![](/docs/img/output_12_1.png)
    



```python
# Remover avisos
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)

# Calcular o percentual de Passageiros a Bordo por Sexo
percentual_por_sexo = df['Sex'].value_counts(normalize=True) * 100
# Gráfico: Percentual de Passageiros a Bordo por Sexo
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Sex', palette='Blues_r')
plt.title('Percentual de Passageiros a Bordo por Sexo', fontsize=11, pad=11)
plt.xlabel('Sexo', fontsize=11)
plt.ylabel('Percentual de Passageiros', fontsize=11)
plt.xticks(ticks=[0, 1], labels=['Masculino', 'Feminino'])  # Definindo os rótulos do eixo x
# Adicionar texto com os percentuais no gráfico
for i, percentual in enumerate(percentual_por_sexo):
    plt.text(i, percentual + 1, f'{percentual:.2f}%', ha='center')
plt.show()

# Calcular o percentual de Sobreviventes por Sexo
percentual_por_sexo = df.groupby('Sex')['Survived'].mean() * 100
# Criar um DataFrame com os dados
quant_sex = pd.DataFrame({'Sexo': percentual_por_sexo.index, 'Taxa de Sobreviventes por Sexo em %': percentual_por_sexo.values})
# Plotar o gráfico
plt.figure(figsize=(6, 4))
ax = sns.barplot(x='Sexo', y='Taxa de Sobreviventes por Sexo em %', data=quant_sex, palette='Blues_r')
ax.set_title("Percentual de Sobreviventes por Sexo", fontsize=11, pad=11)
ax.set_xlabel('Sexo', fontsize=11)
ax.set_ylabel('Taxa de Sobreviventes em %', fontsize=11)
ax.set_xticklabels(labels=['Masculino', 'Feminino'])
plt.tight_layout()
# Adicionar texto com os percentuais no gráfico
for i, percentual in enumerate(percentual_por_sexo):
    plt.text(i, percentual + 1, f'{percentual:.2f}%', ha='center')
plt.show()

# Gráfico: Distribuição da sobrevivência por sexo
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Survived', hue='Sex')
plt.title('Distribuição da Sobrevivência por Sexo')
plt.xlabel('Sobreviveu?')
plt.ylabel('Contagem')
plt.show()
```


    
![](/docs/img/output_13_0.png)
    



    
![](/docs/img/output_13_1.png)
    



    
![](/docs/img/output_13_2.png)
    



```python
# Desative os avisos temporariamente
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Gráfico: Distribuição da sobrevivência por idade
plt.figure(figsize=(6, 4))
sns.histplot(data=df, x='Age', hue='Survived', kde=True)
plt.title('Distribuição da Sobrevivência por Idade')
plt.xlabel('Idade')
plt.ylabel('Contagem')
plt.show()
```


    
![](/docs/img/output_14_0.png)
    



```python
# Gráfico: Distribuição da sobrevivência por classe
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Survived', hue='Pclass')
plt.title('Distribuição da Sobrevivência por Classe')
plt.xlabel('Sobreviveu?')
plt.ylabel('Contagem')
plt.show()
```


    
![](/docs/img/output_15_0.png)
    



```python
# Gráfico: Distribuição da sobrevivência por dependentes
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Survived', hue='SibSp')
plt.title('Distribuição da Sobrevivência por Dependentes-Irmãos/Cônjuge')
plt.xlabel('Sobreviveu?')
plt.ylabel('Contagem')
plt.show()
```


    
![](/docs/img/output_16_0.png)
    



```python
# Gráfico: Distribuição da sobrevivência por dependentes
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Survived', hue='Parch')
plt.title('Distribuição da Sobrevivência por Dependentes-Pais/Filhos')
plt.xlabel('Sobreviveu?')
plt.ylabel('Contagem')
plt.show()
```


    
![](/docs/img/output_17_0.png)
    



```python
# Calcula as estatísticas descritivas
statistics = df.describe(include='all')

# Imprimir as estatísticas descritivas em uma tabela formatada
print("Estatísticas Descritivas:")
print(tabulate(statistics, headers='keys', tablefmt='fancy_grid'))
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
survival_rate_Sex = df.groupby('Sex')['Survived'].mean()
survival_rate_Pclass = df.groupby('Pclass')['Survived'].mean()
survival_rate_Age = df.groupby(pd.cut(df['Age'], bins=[0, 18, 30, 50, 100]))['Survived'].mean()
survival_rate_SibSp = df.groupby('SibSp')['Survived'].mean()
survival_rate_Parch = df.groupby('Parch')['Survived'].mean()

# Memória de Cálculo
print("Taxa de Sobrevivência por Sexo:")
print(tabulate(survival_rate_Sex.reset_index(), headers='keys', tablefmt='fancy_grid'))
print("\nTaxa de Sobrevivência por Classe Socioeconômica:")
print(tabulate(survival_rate_Pclass.reset_index(), headers='keys', tablefmt='fancy_grid'))
print("\nTaxa de Sobrevivência por Faixa Etária:")
print(tabulate(survival_rate_Age.reset_index(), headers='keys', tablefmt='fancy_grid'))
print("\nTaxa de Sobrevivência por Dependentes-Irmãos/Cônjuge:")
print(tabulate(survival_rate_SibSp.reset_index(), headers='keys', tablefmt='fancy_grid'))
print("Taxa de Sobrevivência por Dependentes-Pais/Filhos:")
print(tabulate(survival_rate_Parch.reset_index(), headers='keys', tablefmt='fancy_grid'))

# Fórmulas
print("\nFórmulas Utilizadas:")
print("Taxa de Sobrevivência por Sexo: (Número de Sobreviventes do Sexo / Número Total de Passageiros do Sexo)")
print("Taxa de Sobrevivência por Classe Socioeconômica: (Número de Sobreviventes da Classe / Número Total de Passageiros da Classe)")
print("Taxa de Sobrevivência por Faixa Etária: (Número de Sobreviventes na Faixa Etária / Número Total de Passageiros na Faixa Etária)")
print("Taxa de Sobrevivência por Faixa Etária: (Número de Sobreviventes de Dependentes-Irmãos/Cônjuge / Número Total de Passageiros Dependentes-Irmãos/Cônjuge)")
print("Taxa de Sobrevivência por Faixa Etária: (Número de Sobreviventes de Dependentes-Pais/Filhos / Número Total de Passageiros Dependentes-Pais/Filhos)")

```

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
    
    Taxa de Sobrevivência por Dependentes-Irmãos/Cônjuge:
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
    Taxa de Sobrevivência por Dependentes-Pais/Filhos:
    ╒════╤═════════╤════════════╕
    │    │   Parch │   Survived │
    ╞════╪═════════╪════════════╡
    │  0 │       0 │   0.343658 │
    ├────┼─────────┼────────────┤
    │  1 │       1 │   0.550847 │
    ├────┼─────────┼────────────┤
    │  2 │       2 │   0.5      │
    ├────┼─────────┼────────────┤
    │  3 │       3 │   0.6      │
    ├────┼─────────┼────────────┤
    │  4 │       4 │   0        │
    ├────┼─────────┼────────────┤
    │  5 │       5 │   0.2      │
    ├────┼─────────┼────────────┤
    │  6 │       6 │   0        │
    ╘════╧═════════╧════════════╛
    
    Fórmulas Utilizadas:
    Taxa de Sobrevivência por Sexo: (Número de Sobreviventes do Sexo / Número Total de Passageiros do Sexo)
    Taxa de Sobrevivência por Classe Socioeconômica: (Número de Sobreviventes da Classe / Número Total de Passageiros da Classe)
    Taxa de Sobrevivência por Faixa Etária: (Número de Sobreviventes na Faixa Etária / Número Total de Passageiros na Faixa Etária)
    Taxa de Sobrevivência por Faixa Etária: (Número de Sobreviventes de Dependentes-Irmãos/Cônjuge / Número Total de Passageiros Dependentes-Irmãos/Cônjuge)
    Taxa de Sobrevivência por Faixa Etária: (Número de Sobreviventes de Dependentes-Pais/Filhos / Número Total de Passageiros Dependentes-Pais/Filhos)
    


```python
###### Convertendo infinitos para NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Matriz de dispersão
plt.figure(figsize=(4, 2))
sns.pairplot(df, x_vars=['Sex', 'Parch', 'SibSp', 'Age', 'Pclass'], y_vars=['Sex', 'Parch', 'SibSp', 'Age', 'Pclass'])
```




    <seaborn.axisgrid.PairGrid at 0x7a29c4ed9d20>




    <Figure size 400x200 with 0 Axes>



    
![](/docs/img/output_20_2.png)
    



```python
# matriz de correlação
numeric_df = df.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(12, 12))
sns.heatmap(numeric_df.corr(), annot=True, cmap='Oranges', fmt='.2f')
```




    <Axes: >




    
![](/docs/img/output_21_1.png)
    


Este gráfico é uma matriz de correlação, que mostra as correlações entre diferentes variáveis do conjunto de dados do naufrágio do Titanic. Cada célula colorida representa o coeficiente de correlação entre duas variáveis específicas. As cores mais avermelhadas indicam correlação positiva, enquanto as cores mais alaranjadas/amareladas indicam correlação negativa.
 
Algumas observações importantes:
 
1. A variável "PassengerId" tem correlação próxima de 1 consigo mesma, o que é esperado, já que é um identificador único.
 
2. As variáveis "Survived" e "Pclass" mostram uma correlação negativa moderada, sugerindo que passageiros de classes mais altas tiveram maior probabilidade de sobrevivência.
 
3. "Age" mostra correlações negativas fracas com "Survived" e "Pclass", indicando que pessoas mais jovens tinham uma leve vantagem de sobrevivência e tendiam a estar em classes mais altas.
 
4. "Sibsp" (número de irmãos/cônjuges a bordo) tem uma correlação positiva fraca com "Parch" (número de pais/filhos a bordo), sugerindo que famílias maiores viajavam juntas.
 
5. "Fare" tem uma correlação positiva moderada com "Pclass", o que faz sentido, já que passageiros de classes mais altas pagavam tarifas mais altas.
 
Em resumo, este gráfico de correlações pode fornecer insights iniciais sobre os relacionamentos entre as variáveis do conjunto de dados do Titanic e orientar uma análise mais aprofundada.
