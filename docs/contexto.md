# Introdução

O RMS Titanic foi um navio de passageiros britânico operado pela White Star Line e construído pelos estaleiros da Harland and Wolff, em Belfast. Segunda embarcação da Classe Olympic de transatlânticos, depois do RMS Olympic e seguido pelo HMHS Britannic, foi projetado pelos engenheiros navais Alexander Carlisle e Thomas Andrews. Sua construção começou em março de 1909 e seu lançamento ao mar ocorreu em maio de 1911. O Titanic foi pensado para ser o navio mais luxuoso e mais seguro de sua época, gerando lendas que era supostamente "inafundável".

A embarcação partiu em sua viagem inaugural de Southampton com destino a Nova Iorque em 10 de abril de 1912, no caminho passando em Cherbourg-Octeville, na França, e por Queenstown, na Irlanda. Colidiu com um iceberg na proa do lado direito às 23h40 de 14 de abril, naufragando na madrugada do dia seguinte.
Infelizmente, não havia botes salva-vidas suficientes para todos a bordo, resultando na morte de 1.502 dos 2.224 passageiros e tripulantes. Sendo um dos maiores desastres marítimos em tempos de paz de toda a história.

Este Dataset contém informações sobre os passageiros a bordo, permitindo uma análise detalhada dos fatores que influenciaram suas chances de sobrevivência. O público-alvo deste projeto são pesquisadores, entusiastas de aprendizado de máquina e qualquer pessoa interessada em entender melhor os padrões de sobrevivência no desastre do Titanic.

## Problema

Previsibilidade de sobrevivência dos passageiros do Titanic com base em uma gama de variáveis disponíveis no conjunto de dados, como idade, sexo, classe socioeconômica, número de familiares a bordo, entre outros. Essa previsão é relevante não apenas do ponto de vista histórico, mas também pode fornecer insights significativos sobre os fatores que influenciaram as chances de sobrevivência durante o desastre do Titanic.

Ao compreender os padrões de sobrevivência, podemos responder a perguntas importantes, como: Quais grupos demográficos tiveram maiores chances de sobreviver? Qual foi o impacto da classe socioeconômica na probabilidade de sobrevivência? Existem diferenças significativas nas taxas de sobrevivência entre homens e mulheres? E como outras variáveis, como idade e tamanho da família, influenciaram as chances de sobrevivência?

Além disso, esta análise pode ter implicações mais amplas, incluindo a identificação de medidas de segurança mais eficazes em viagens marítimas e a compreensão dos comportamentos humanos em situações de crise. Portanto, o problema vai além da mera previsão de sobrevivência; ele busca fornecer insights valiosos que podem informar políticas e práticas de segurança em contextos similares.

> **Links Úteis**:
> - [Objetivos, Problema de pesquisa e Justificativa](https://medium.com/@versioparole/objetivos-problema-de-pesquisa-e-justificativa-c98c8233b9c3)
> - [Matriz Certezas, Suposições e Dúvidas](https://medium.com/educa%C3%A7%C3%A3o-fora-da-caixa/matriz-certezas-suposi%C3%A7%C3%B5es-e-d%C3%BAvidas-fa2263633655)
> - [Brainstorming](https://www.euax.com.br/2018/09/brainstorming/)

## Questão de pesquisa

Qual é a influência da idade, parentes a bordo, sexo, classe socioeconômica e número de parentes a bordo em suas chances de sobrevivência no naufrágio do Titanic?

> **Links Úteis**:
> - [Questão de pesquisa](https://www.enago.com.br/academy/how-to-develop-good-research-question-types-examples/)
> - [Problema de pesquisa](https://blog.even3.com.br/problema-de-pesquisa/)

## Objetivos preliminares

Objetivo Geral: 
Experimentar modelos de aprendizado de máquina adequados para prever a sobrevivência dos passageiros do Titanic.

Objetivos Específicos:

1. Avaliar o impacto da idade e sexo na probabilidade de sobrevivência.
2. Analisar a relação entre classe socioeconômica e chances de sobrevivência.

Ao avaliar o impacto da idade e sexo na probabilidade de sobrevivência, o projeto busca entender se os fatores demográficos têm influência na sobrevivência dos passageiros. A análise deve fornecer informações sobre as taxas de sobrevivência para diferentes grupos de idade e sexo, permitindo identificar quais grupos tiveram maiores chances de sobrevivência.

A segunda meta é analisar a relação entre classe socioeconômica e chances de sobrevivência. Neste aspecto, o projeto busca entender se os passageiros de classes mais altas tiveram maiores chances de sobrevivência, considerando o acesso aos botes salva-vidas e outras fatores relacionados à classe socioeconômica.

Estes objetivos específicos são importantes para compreender os padrões de sobrevivência no desastre do Titanic e fornecer insights valiosos que podem informar políticas e práticas de segurança em contextos similares.
 
> **Links Úteis**:
> - [Objetivo geral e objetivo específico: como fazer e quais verbos utilizar](https://blog.mettzer.com/diferenca-entre-objetivo-geral-e-objetivo-especifico/)

## Justificativa

A análise da sobrevivência no desastre do Titanic pode oferecer insights históricos e ajudar a entender padrões de comportamento humano em situações de crise. Além disso, a identificação dos principais fatores que influenciam a sobrevivência pode informar medidas de segurança em viagens marítimas e outros contextos similares. Dados reais sobre desastres como o do Titanic são raros e valiosos para estudos acadêmicos e práticos.

Existem vários estudos que analisam a influência do comportamento humano em situações de crise, como o desastre do Titanic. Por exemplo, um estudo publicado no Journal of Quantitative Criminology analisa o papel da idade e sexo na sobrevivência dos passageiros do Titanic. Outro estudo, publicado no Journal of Emerging Technologies And Innovative Research, analisa a influência da classe socioeconômica na sobrevivência dos passageiros.

Além disso, um estudo realizado pela UC Berkeley analisa o comportamento das pessoas durante o desastre do Titanic e fornece informações que podem ajudar a melhorar as medidas de segurança em situações de emergência.

Esses estudos podem fornecer insights valiosos para entender os padrões de comportamento humano em situações de crise e informar medidas de segurança em viagens marítimas e outros contextos similares.

## Público-Alvo

Inclui pesquisadores acadêmicos interessados em estudos históricos e comportamentais, profissionais de segurança em viagens marítimas e entusiastas de aprendizado de máquina. Os usuários podem variar em seu nível de conhecimento técnico, desde iniciantes até especialistas em análise de dados.
Um possivel viajante.

> **Links Úteis**:
> - [Público-alvo](https://blog.hotmart.com/pt-br/publico-alvo/)
> - [Como definir o público alvo](https://exame.com/pme/5-dicas-essenciais-para-definir-o-publico-alvo-do-seu-negocio/)
> - [Público-alvo: o que é, tipos, como definir seu público e exemplos](https://klickpages.com.br/blog/publico-alvo-o-que-e/)
> - [Qual a diferença entre público-alvo e persona?](https://rockcontent.com/blog/diferenca-publico-alvo-e-persona/)

## Estado da arte

A análise do desastre do Titanic através de algoritmos de machine learning já foi realizada em diversos estudos. O artigo "Analysis of Titanic Disaster using Machine Learning Algorithms" utiliza algoritmos de aprendizado de máquina, como Random Forest, XGBoost e Neural Networks, para prever a sobrevivência dos passageiros do Titanic. O estudo obteve acurácia de 84,1% usando o algoritmo Random Forest.

Outro estudo, "Survival prediction of Titanic disaster using machine learning: Re-visit with Neural Network Approach", utiliza uma abordagem diferente, aplicando redes neurais profundas para prever a sobrevivência dos passageiros do Titanic. O estudo obteve uma acurácia de 83,7% usando redes neurais profundas.

Além disso, o artigo "A Comparative Study on Machine Learning Techniques Using Titanic Dataset" compara a eficácia de diferentes algoritmos de classificação, como Decision Trees, Random Forest, Logistic Regression, e K-Nearest Neighbors, na previsão da sobrevivência dos passageiros do Titanic. O estudo obteve uma acurácia de 84,8% usando o algoritmo Random Forest.

Esses estudos demonstram a eficácia de diferentes algoritmos de machine learning na previsão da sobrevivência dos passageiros do Titanic. No entanto, ainda há espaço para melhorias e aplicação de novas técnicas de aprendizado de máquina para obter mais insights sobre os fatores que influenciaram a sobrevivência dos passageiros do Titanic.

# Descrição do _dataset_ selecionado

O Dataset consiste de um conjunto de dados relacionados ao RMS Titanic e seu acidente no ano de 1912, sendo esse fonte para geração de dashboards e reports, seja para estudos acadêmicos ou profissionais, sendo base para a aplicação de machine learning.

**Origem do Dataset:** https://www.kaggle.com/

**Glossário:** o termo "passageiro" se aplica ao passageiro ou tribulante que estava à bordo do RMS Titanic

**Variáveis do Dataset:**
- **PassengerID:** número de identificação único atribuído a cada registro do Dataset
- **Survived:** descreve se o passageiro sobreviu, representado pelos valores 0 = No e 1 = Yes
- **Pclass:** descreve a classe do ticket do passageiro, representado pelos valores 1 = 1º Classe; 2 = 2º Classe; 3 = 3º Classe
- **Name:** descreve o nome do passageiro
- **Sex:** descreve o gênero sexual do passageiro, representado por male ou female
- **Age:** descreve a idade do passageiro, sendo uma variável numérica do tipo inteiro
- **SibSp:** descreve a quantidade irmãos ou cônjuge à bordo do respectivo passageiro, sendo representado por variável numérica do tipo inteiro
- **Parch:** descreve a quantidade de pais ou filhos à bordo do respectivo passageiro, sendo representado por uma variável numérica do tipo inteiro
- **Ticket:** descreve o número do ticket do passageiro
- **Fare:** descreve a tarifa paga pelo passaageiro, sendo representado por uma variável numérica do tipo decimal

**Dados ausentes:**
Conforme demonstrado anteriormente, esse Dataset é representado uma limitação de variáveis que permitem obter alguns insights específicos, como: compreender a faixa etária dos sobrevientes, comprender o percentual de sobrevientes por gênero sexual, entre vários outros insights. Entretanto, existem dados ausentes  que permitiram obter outros tipo de análises. Por exemplo, podemos citar a ausência de informações referentes ao posto de trabalho de cada tribulante, qual a cidadania de cada passageiro, número de botes à bordo, capacidade dos botes, etc.

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

> **Links Úteis**:
> - [Modelo do Canvas Analítico](https://github.com/ICEI-PUC-Minas-PMV-SI/PesquisaExperimentacao-Template/blob/main/help/Software-Analtics-Canvas-v1.0.pdf)

# Referências

IBRAHIM, Abdullahi Adinoyi. Analysis of Titanic Disaster using Machine Learning Algorithms. Engineering Letters, v. 28, n. 4, p. EL_28_4_22, 2021. Disponível em: <https://www.researchgate.net/publication/353352089_Analysis_of_Titanic_Disaster_using_Machine_Learning_Algorithms>.

KAVYA, N. C.; SRINIVASULU, M. Survival prediction of Titanic disaster using machine learning: Re-visit with Neural Network Approach. Journal of Emerging Technologies and Innovative Research (JETIR), v. 9, n. 9, p. b320, 2022. Disponível em: <https://www.jetir.org/papers/JETIR2209140.pdf>.

FREY, Bruno S.; SAVAGE, David A.; TORGLER, Benno. Surviving the Titanic Disaster: Economic, Natural and Social Determinants. eScholarship.org, 2009. p. 7. Disponível em: <https://escholarship.org/content/qt6h24b1vt/qt6h24b1vt.pdf>.

EKINCI, E.; OMURCA, S. İ.; ACUN, N.  [2018]. A Comparative Study on Machine Learning Techniques using Titanic Dataset. In: 7th International Conference on Advanced Technologies (ICAT'18), pp. 1-X, April 28-May 1, Antalya/TURKEY. Disponível em: <https://www.researchgate.net/publication/324909545_A_Comparative_Study_on_Machine_Learning_Techniques_Using_Titanic_Dataset>.


> **Links Úteis**:
> - [Padrão ABNT PUC Minas](https://portal.pucminas.br/biblioteca/index_padrao.php?pagina=5886)
