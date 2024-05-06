# Pergunta orientada a dados

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

* Limpeza de Dados: trate valores ausentes: decida como lidar com dados faltantes, seja removendo linhas, preenchendo com médias, medianas ou usando métodos mais avançados; remova _outliers_: identifique e trate valores que se desviam significativamente da maioria dos dados.

* Transformação de Dados: normalize/padronize: torne os dados comparáveis, normalizando ou padronizando os valores para uma escala específica; codifique variáveis categóricas: converta variáveis categóricas em uma forma numérica, usando técnicas como _one-hot encoding_.

* _Feature Engineering_: crie novos atributos que possam ser mais informativos para o modelo; selecione características relevantes e descarte as menos importantes.

* Tratamento de dados desbalanceados: se as classes de interesse forem desbalanceadas, considere técnicas como _oversampling_, _undersampling_ ou o uso de algoritmos que lidam naturalmente com desbalanceamento.

* Separação de dados: divida os dados em conjuntos de treinamento, validação e teste para avaliar o desempenho do modelo de maneira adequada.
  
* Manuseio de Dados Temporais: se lidar com dados temporais, considere a ordenação adequada e técnicas específicas para esse tipo de dado.
  
* Redução de Dimensionalidade: aplique técnicas como PCA (Análise de Componentes Principais) se a dimensionalidade dos dados for muito alta.

* Validação Cruzada: utilize validação cruzada para avaliar o desempenho do modelo de forma mais robusta.

* Monitoramento Contínuo: atualize e adapte o pré-processamento conforme necessário ao longo do tempo, especialmente se os dados ou as condições do problema mudarem.

* Entre outras....

Avalie quais etapas são importantes para o contexto dos dados que você está trabalhando, pois a qualidade dos dados e a eficácia do pré-processamento desempenham um papel fundamental no sucesso de modelo(s) de aprendizado de máquina. É importante entender o contexto do problema e ajustar as etapas de preparação de dados de acordo com as necessidades específicas de cada projeto.

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
