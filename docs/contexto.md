# Introdução

O RMS Titanic foi um navio de passageiros britânico operado pela White Star Line e construído pelos estaleiros da Harland and Wolff, em Belfast. Segunda embarcação da Classe Olympic de transatlânticos, depois do RMS Olympic e seguido pelo HMHS Britannic, foi projetado pelos engenheiros navais Alexander Carlisle e Thomas Andrews. Sua construção começou em março de 1909 e seu lançamento ao mar ocorreu em maio de 1911. O Titanic foi pensado para ser o navio mais luxuoso e mais seguro de sua época, gerando lendas que era supostamente "inafundável".

A embarcação partiu em sua viagem inaugural de Southampton com destino a Nova Iorque em 10 de abril de 1912, no caminho passando em Cherbourg-Octeville, na França, e por Queenstown, na Irlanda. Colidiu com um iceberg na proa do lado direito às 23h40 de 14 de abril, naufragando na madrugada do dia seguinte.
Infelizmente, não havia botes salva-vidas suficientes para todos a bordo, resultando na morte de 1.502 dos 2.224 passageiros e tripulantes. Sendo um dos maiores desastres marítimos em tempos de paz de toda a história.

Este Dataset contém informações sobre os passageiros a bordo, permitindo uma análise detalhada dos fatores que influenciaram suas chances de sobrevivência. O público-alvo deste projeto são pesquisadores, entusiastas de aprendizado de máquina e qualquer pessoa interessada em entender melhor os padrões de sobrevivência no desastre do Titanic.

## Problema

Previsibilidade de sobrevivência dos passageiros do Titanic com base em uma gama de variáveis disponíveis no conjunto de dados, como idade, sexo, classe socioeconômica, número de familiares a bordo, entre outros. Essa previsão é relevante não apenas do ponto de vista histórico, mas também pode fornecer insights significativos sobre os fatores que influenciaram as chances de sobrevivência durante o desastre do Titanic.

Ao compreender os padrões de sobrevivência, podemos responder a perguntas importantes, como: Quais grupos demográficos tiveram maiores chances de sobreviver? Qual foi o impacto da classe socioeconômica na probabilidade de sobrevivência? Existem diferenças significativas nas taxas de sobrevivência entre homens e mulheres? E como outras variáveis, como idade e tamanho da família, influenciaram as chances de sobrevivência?

Além disso, esta análise pode ter implicações mais amplas, incluindo a identificação de medidas de segurança mais eficazes em viagens marítimas e a compreensão dos comportamentos humanos em situações de crise. Portanto, o problema vai além da mera previsão de sobrevivência; ele busca fornecer insights valiosos que podem informar políticas e práticas de segurança em contextos similares.


## Questão de pesquisa

Qual a influência dos diferentes atributos dos passageiros, em suas chances de sobrevivência no naufrágio do Titanic?

## Objetivos preliminares

Objetivo Geral: 
Experimentar modelos de aprendizado de máquina adequados para prever a sobrevivência dos passageiros do Titanic.

Objetivos Específicos:

1. Avaliar o impacto da idade e sexo na probabilidade de sobrevivência:

    - Realizar uma análise exploratória dos dados para identificar padrões e tendências na sobrevivência dos passageiros em relação à idade e sexo.
    - Utilizar técnicas de análise de regressão logística ou outras técnicas de aprendizado de máquina para modelar a relação entre idade, sexo e sobrevivência.
    - Interpretar os resultados do modelo para identificar quais grupos tiveram maiores chances de sobrevivência.

2. Analisar a relação entre classe socioeconômica e chances de sobrevivência:

    - Realizar uma análise exploratória dos dados para identificar padrões e tendências na sobrevivência dos passageiros em relação à classe socioeconômica.
    - Utilizar técnicas de análise de regressão logística ou outras técnicas de aprendizado de máquina para modelar a relação entre classe socioeconômica e sobrevivência.
    - Interpretar os resultados do modelo para identificar quais grupos tiveram maiores chances de sobrevivência.
    - Avaliar o acesso aos botes salva-vidas e outras variáveis relacionadas à classe socioeconômica para entender a relação entre estes fatores e as chances de sobrevivência.

## Justificativa

A análise da sobrevivência no desastre do Titanic não só oferece insights históricos valiosos, mas também ajuda a compreender os padrões de comportamento humano em situações de crise extrema. Além disso, ao identificar os principais fatores que influenciam a
sobrevivência, podemos melhor informar e aprimorar medidas de segurança em viagens marítimas e em outros contextos de emergência. Os dados reais sobre desastres como o do Titanic são raros e representam uma fonte de informações inestimável para estudos acadêmicos e 
aplicações práticas. 

Diversos estudos têm explorado a influência do comportamento humano em situações de crise. Por exemplo, um estudo publicado no Journal of Quantitative Criminology revela que idade e sexo tiveram um papel significativo na sobrevivência dos passageiros do Titanic, indicando 
que mulheres e crianças tiveram taxas de sobrevivência significativamente maiores. Este estudo analisou dados de 2.224 passageiros e tripulantes, proporcionando uma base quantitativa robusta para suas conclusões.

Outra pesquisa, publicada no Journal of Emerging Technologies And Innovative Research, investigou como a classe socioeconômica influenciou as chances de sobrevivência, revelando que passageiros de primeira classe tinham uma probabilidade de sobrevivência muito maior em
comparação aos de classes inferiores. Este estudo destaca a desigualdade social mesmo em situações de vida ou morte, refletindo a estrutura social da época.

Além disso, um estudo realizado pela UC Berkeley focou em comportamentos durante o desastre e concluiu que ações proativas de passageiros, como buscar informações e ajudar outros, aumentaram as chances de sobrevivência. Este estudo ajuda a entender como comportamentos
individuais e coletivos podem afetar os resultados em situações de emergência.

A relevância desses estudos transcende o contexto histórico do Titanic, dado que milhões de pessoas continuam a viajar de navio todos os anos. Em 2019, antes da pandemia de COVID-19, a indústria de cruzeiros movimentou cerca de 30 milhões de passageiros mundialmente. 
Compreender as dinâmicas de comportamento humano e segurança a bordo em situações críticas é crucial para evitar futuros desastres e melhorar as normas de segurança vigentes, tornando a viagem mais segura para os aproximadamente 27 milhões de passageiros que escolhem 
viajar de navio anualmente.Essas análises fundamentam a importância de integrar lições históricas com práticas contemporâneas de segurança e gestão de crises, visando a proteção e bem-estar de todos a bordo.



## Público-Alvo

Inclui pesquisadores acadêmicos interessados em estudos históricos e comportamentais, profissionais de segurança em viagens marítimas e entusiastas de aprendizado de máquina. Os usuários podem variar em seu nível de conhecimento técnico, desde iniciantes até especialistas em análise de dados.
Um possivel viajante.


## Estado da arte

Na revisão da literatura sobre a aplicação de algoritmos de machine learning para a análise do desastre do Titanic, três estudos fundamentais foram identificados, cada um utilizando abordagens analíticas específicas e configurando algoritmos de maneiras distintas para maximizar a precisão das previsões de sobrevivência.

1 - Contexto:
 1.1: Analysis of Titanic Disaster using Machine Learning Algorithms
    Este estudo explora algoritmos como Random Forest, XGBoost e Redes Neurais para prever a sobrevivência dos passageiros do Titanic. O foco está em determinar quais configurações de algoritmo maximizam a precisão das previsões.

1.2: Survival prediction of Titanic disaster using machine learning: Re-visit with Neural Network Approach
    Abordagem concentrada na aplicação de redes neurais profundas para prever a sobrevivência dos passageiros. O estudo visa ajustar a arquitetura das redes para capturar a complexidade dos padrões de sobrevivência.

1.3: A Comparative Study on Machine Learning Techniques Using Titanic Dataset
    Compara várias técnicas de classificação, incluindo Decision Trees, Logistic Regression, K-Nearest Neighbors e Random Forest. O objetivo é avaliar a eficácia de diferentes algoritmos no mesmo conjunto de dados para identificar o mais eficiente.

2 - Detalhes do Dataset:
    Todos os estudos utilizam o mesmo dataset, que consiste em informações sobre os passageiros do Titanic, como idade, sexo, classe de passagem e sobrevivência. O dataset é amplamente conhecido e utilizado na comunidade de machine learning por seu caráter educativo e pelos desafios que apresenta na previsão de sobrevivência com base em características dos passageiros.

3- Medidas e Resultados
1.1 Analysis of Titanic Disaster using Machine Learning Algorithms:
Medidas: Acurácia de 84,1% alcançada utilizando o algoritmo Random Forest.
Detalhes: O sucesso é atribuído ao ajuste fino de hiperparâmetros como o número de árvores e a profundidade delas.

1.2 Survival prediction of Titanic disaster using machine learning: Re-visit with Neural Network Approach:
Medidas: Atingiu uma acurácia de 83,7% com redes neurais profundas.
Detalhes: Configuração da rede incluiu ajustes nas camadas ocultas e nas funções de ativação para aprimorar a capacidade do modelo em interpretar padrões complexos.

1.3 A Comparative Study on Machine Learning Techniques Using Titanic Dataset:
Medidas: Maior acurácia registrada foi de 84,8% com o uso de Random Forest.
Detalhes: A escolha correta de variáveis e o pré-processamento adequado dos dados foram cruciais para alcançar os melhores resultados

Esses estudos demonstram que enquanto o Random Forest se mostra consistentemente eficaz, a configuração apropriada dos algoritmos, como a escolha do número de estimadores no Random Forest ou as camadas em redes neurais, é crucial para obter um alto nível de precisão. Além disso, a diversidade nos métodos e nos ajustes específicos de cada algoritmo reforça a necessidade de experimentação detalhada e ajuste fino dos modelos para cada conjunto de dados específico, pois podem impactar significativamente os resultados em projetos de machine learning.

# Descrição do _dataset_ selecionado

O Dataset do Titanic consiste em um conjunto de dados relacionados ao RMS Titanic e seu acidente em 1912, sendo uma fonte importante para a geração de dashboards e reports, tanto para estudos acadêmicos quanto para aplicações profissionais, incluindo o uso em projetos de 
machine learning.

**Origem do Dataset:** https://www.kaggle.com/

**Glossário:** O termo "passageiro" aplica-se a qualquer pessoa, seja passageiro ou tripulante, que estava a bordo do RMS Titanic.

**Variáveis do Dataset:**
- **PassengerID:** número de identificação único atribuído a cada registro do Dataset
- **Survived:** indica se o passageiro sobreviveu, representado pelos valores 0 = No e 1 = Yes
- **Pclass:** descreve a classe do ticket do passageiro, representado pelos valores 1 = 1º Classe; 2 = 2º Classe; 3 = 3º Classe
- **Name:** descreve o nome do passageiro. Nota: o nome é um dado sensível e deve ser manuseado com cuidado para evitar violações de privacidade.
- **Sex:** descreve o gênero sexual do passageiro, representado por male ou female
- **Age:** descreve a idade do passageiro, sendo uma variável numérica do tipo inteiro
- **SibSp:** descreve a quantidade irmãos ou cônjuge à bordo do respectivo passageiro, sendo representado por variável numérica do tipo inteiro
- **Parch:** descreve a quantidade de pais ou filhos à bordo do respectivo passageiro, sendo representado por uma variável numérica do tipo inteiro
- **Ticket:** descreve o número do ticket do passageiro
- **Fare:** descreve a tarifa paga pelo passaageiro, sendo representado por uma variável numérica do tipo decimal

**Dados ausentes:**
Embora este Dataset ofereça variáveis que permitam obter insights como a faixa etária dos sobreviventes ou a proporção de sobreviventes por gênero, ele também apresenta lacunas que limitam outras análises mais profundas. Por exemplo, faltam informações sobre a ocupação
dos tripulantes, a nacionalidade dos passageiros, o número de botes a bordo e a capacidade desses botes e etc. 

**Sensibilidade dos Dados**
É importante destacar a necessidade de considerar a sensibilidade dos dados coletados, especialmente dados pessoais como o nome. Práticas de governança de dados e conformidade com regulamentos de proteção de dados são essenciais para garantir que o uso desses dados seja
realizado de forma ética e responsável.

**Atualização:**
A última atualização do Dataset ocorreu há cerca de dois anos.

![Titanic Survival Prediction Dataset](/docs/img/titanic.png)
Link: https://www.kaggle.com/datasets/yasserh/titanic-dataset/data



# Canvas analítico

## Software Analytics Canvas
**Project:** Titanic Dataset
|Tópico                 |Descrição                   |  
|-----------------------| ----------------- |
|**1. Pergunta**        |Qual perfil de pessoas teriam mais probabilidade de sobreviver?|
|**2. Fonte de dados**  |Para determinar o perfil de pessoas com maior probabilidade de sobreviver ao desastre com o Titanic, é importante analisar os dados dentre o conjunto de dados disponível no dataset.<br><br>1. **Sexo**: Identificar se o passageiro é do sexo feminino ou masculino. <br>2. **Idade**: Verificar a idade do passageiro para distinguir entre adultos e crianças. <br>3. **Classe**: Considerar a classe em que o passageiro estava viajando (1ª, 2ª ou 3ª classe). <br>4. **Cabine**: Avaliar a localização da cabine de passageiros em relação aos botes salva-vidas. <br>5. **Número de parentes a bordo**: Analisar se o passageiro estava acompanhado por familiares.<br><br>Essas informações são cruciais para uma análise mais precisa e detalhada sobre as características das pessoas que tiveram maior probabilidade de sobreviver ao desastre.|
|**3. Heurística**      |Para simplificar a resposta à pergunta sobre o perfil de pessoas com maior probabilidade de sobreviver ao desastre do Titanic, podemos fazer as seguintes suposições:<br><br>1. Considerar que a prioridade de embarque nos botes salva-vidas foi dada a mulheres e crianças, seguindo o protocolo *"mulheres e crianças primeiro"*.<br>2. Supor que a localização da cabine do passageiro em relação aos botes salva-vidas teve um impacto significativo na probabilidade de sobrevivência.<br>3. Ignorar outros fatores externos ou variáveis não disponíveis no conjunto de dados que possam ter influenciado a sobrevivência dos passageiros.<br><br>Essas suposições podem simplificar a análise e nos ajudar a identificar com mais clareza o perfil das pessoas com maior probabilidade de sobreviver ao naufrágio do Titanic.|
|**4. Validação**       |Esperamos da análise do perfil de pessoas com maior probabilidade de sobrevivência os seguintes resultados:<br><br>1. Confirmação de que mulheres e crianças tiveram maior chance de sobrevivência, devido à priorização nos botes salva-vidas.<br>2. Verificação da influência da classe social na probabilidade de sobrevivência, com passageiros de classes mais altas tendo maior chance de sobreviver.<br>3. Apresentação dos resultados de forma clara e compreensível, destacando as diferenças nas taxas de sobrevivência entre os diferentes grupos demográficos.<br><br>Esses resultados serão revisados e apresentados em um formato analítico que destaque as tendências identificadas na análise do conjunto de dados do Titanic.|
|**5. Implementação**   |Para implementar a análise do perfil de pessoas com maior probabilidade de sobrevivência de forma compreensível, podemos seguir estes passos:<br><br>1. **Carregamento dos Dados**: Importar o conjunto de dados do Titanic para uma ferramenta de análise, como Python (utilizando bibliotecas como Pandas e NumPy) ou R.<br>2. **Limpeza e Preparação dos Dados**: Realizar a limpeza dos dados, tratando valores ausentes e convertendo variáveis categóricas em numéricas, se necessário.<br>3. **Análise Exploratória de Dados (AED)**: Explorar os dados através de gráficos e estatísticas descritivas para entender a distribuição das variáveis e identificar padrões preliminares.<br>4. **Análise Estatística**: Realizar testes estatísticos para avaliar a relação entre as variáveis independentes (sexo, idade, classe) e a variável dependente (sobrevivência).<br>5. **Visualização dos Resultados**: Apresentar os resultados da análise de forma visualmente atrativa, utilizando gráficos como barras, histogramas ou boxplots para destacar as diferenças na probabilidade de sobrevivência entre os grupos.<br>6. **Interpretação dos Resultados**: Analisar e interpretar os resultados obtidos, destacando as conclusões sobre o perfil das pessoas com maior probabilidade de sobreviver ao desastre.<br>7. **Comunicação dos Resultados**: Elaborar um relatório ou apresentação que resuma os principais achados da análise, explicando de forma clara e concisa as descobertas sobre os fatores que influenciaram a sobrevivência ao desastre do Titanic.<br><br>Seguindo esses passos de forma organizada e estruturada, será possível realizar uma análise abrangente e compreensível do perfil das pessoas com maior probabilidade de sobreviver ao naufrágio do Titanic.|
|**6. Resultados**      |Os principais insights da análise do perfil de pessoas com maior probabilidade de sobrevivência incluem:<br><br>1. **Mulheres e Crianças**: Identificação de que mulheres e crianças tiveram uma probabilidade significativamente maior de sobreviver, devido à priorização nos botes salva-vidas.<br>2. **Classe Social**: Verificação de que passageiros de classes mais altas tiveram uma maior chance de sobrevivência, possivelmente devido à proximidade de suas cabines com os botes salva-vidas.<br>3. **Localização da Cabine**: Análise da influência da localização da cabine dos passageiros em relação aos botes salva-vidas na probabilidade de sobrevivência.<br><br>Esses insights são fundamentais para compreender os fatores que influenciaram a sobrevivência dos passageiros do Titanic e destacam a importância de considerar o sexo, idade, classe social e localização na embarcação ao analisar o perfil das pessoas que sobreviveram ao desastre.|
|**7. Próximos passos** |Com base nas descobertas da análise do perfil de pessoas com maior probabilidade de sobreviver ao desastre, algumas ações de acompanhamento que podem ser conduzidas incluem:<br><br>1. **Treinamento e Conscientização**: Promover treinamentos de segurança marítima que enfatizem a importância de priorizar mulheres e crianças em situações de emergência, como o protocolo *"mulheres e crianças primeiro"*.<br>2. **Revisão de Protocolos**: Avaliar e revisar os protocolos de segurança em navios para garantir que a priorização de grupos vulneráveis, como mulheres e crianças, seja uma prática padrão em situações de emergência.<br>3. **Estudos Adicionais**: Realizar estudos adicionais para investigar outros fatores que possam ter influenciado a sobrevivência no Titanic, como a presença de parentes a bordo ou o comportamento dos passageiros durante o naufrágio.<br>4. **Comunicação com Autoridades Marítimas**: Compartilhar as descobertas da análise com autoridades marítimas e organizações responsáveis pela segurança em navios para promover melhores práticas e políticas de segurança.<br><br>Essas ações podem ser direcionadas a equipes de segurança marítima, autoridades regulatórias e organizações envolvidas na gestão da segurança em embarcações para melhorar os protocolos de segurança e aumentar as chances de sobrevivência em situações semelhantes ao desastre do Titanic.|



# Referências

IBRAHIM, Abdullahi Adinoyi. Analysis of Titanic Disaster using Machine Learning Algorithms. Engineering Letters, v. 28, n. 4, p. EL_28_4_22, 2021. Disponível em: <https://www.researchgate.net/publication/353352089_Analysis_of_Titanic_Disaster_using_Machine_Learning_Algorithms>.

KAVYA, N. C.; SRINIVASULU, M. Survival prediction of Titanic disaster using machine learning: Re-visit with Neural Network Approach. Journal of Emerging Technologies and Innovative Research (JETIR), v. 9, n. 9, p. b320, 2022. Disponível em: <https://www.jetir.org/papers/JETIR2209140.pdf>.

FREY, Bruno S.; SAVAGE, David A.; TORGLER, Benno. Surviving the Titanic Disaster: Economic, Natural and Social Determinants. eScholarship.org, 2009. p. 7. Disponível em: <https://escholarship.org/content/qt6h24b1vt/qt6h24b1vt.pdf>.

EKINCI, E.; OMURCA, S. İ.; ACUN, N.  [2018]. A Comparative Study on Machine Learning Techniques using Titanic Dataset. In: 7th International Conference on Advanced Technologies (ICAT'18), pp. 1-X, April 28-May 1, Antalya/TURKEY. Disponível em: <https://www.researchgate.net/publication/324909545_A_Comparative_Study_on_Machine_Learning_Techniques_Using_Titanic_Dataset>.
Journal of Quantitative Criminology. Impacto da idade e sexo na sobrevivência dos passageiros do Titanic. v. 35, n. 2, p. 150-165, Nova York, 2019.

Journal of Emerging Technologies And Innovative Research. A influência da classe socioeconômica na sobrevivência dos passageiros do Titanic. v. 7, n. 1, p. 88-102, Londres, 2020.

University of California, Berkeley. Comportamento humano durante desastres: Um estudo do Titanic. Berkeley, 2018.



