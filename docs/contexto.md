# Introdução

Teste

O RMS Titanic foi um navio de passageiros britânico operado pela White Star Line e construído pelos estaleiros da Harland and Wolff, em Belfast. Segunda embarcação da Classe Olympic de transatlânticos, depois do RMS Olympic e seguido pelo HMHS Britannic, foi projetado pelos engenheiros navais Alexander Carlisle e Thomas Andrews. Sua construção começou em março de 1909 e seu lançamento ao mar ocorreu em maio de 1911. O Titanic foi pensado para ser o navio mais luxuoso e mais seguro de sua época, gerando lendas que era supostamente "inafundável".

A embarcação partiu em sua viagem inaugural de Southampton com destino a Nova Iorque em 10 de abril de 1912, no caminho passando em Cherbourg-Octeville, na França, e por Queenstown, na Irlanda. Colidiu com um iceberg na proa do lado direito às 23h40 de 14 de abril, naufragando na madrugada do dia seguinte.
Infelizmente, não havia botes salva-vidas suficientes para todos a bordo, resultando na morte de 1.502 dos 2.224 passageiros e tripulantes. Sendo um dos maiores desastres marítimos em tempos de paz de toda a história.

Este Dataset contém informações sobre os passageiros a bordo, permitindo uma análise detalhada dos fatores que influenciaram suas chances de sobrevivência. O público-alvo deste projeto são pesquisadores, entusiastas de aprendizado de máquina e qualquer pessoa interessada em entender melhor os padrões de sobrevivência no desastre do Titanic.

## Problema

Previsibilidade de sobrevivência dos passageiros do Titanic com base em uma variedade de variáveis disponíveis no conjunto de dados, como idade, sexo, classe socioeconômica, número de familiares a bordo, entre outros. Essa previsão é relevante não apenas do ponto de vista histórico, mas também pode fornecer insights significativos sobre os fatores que influenciaram as chances de sobrevivência durante o desastre do Titanic.

Ao compreender os padrões de sobrevivência, podemos responder a perguntas importantes, como: Quais grupos demográficos tiveram maiores chances de sobreviver? Qual foi o impacto da classe socioeconômica na probabilidade de sobrevivência? Existem diferenças significativas nas taxas de sobrevivência entre homens e mulheres? E como outras variáveis, como idade e tamanho da família, influenciaram as chances de sobrevivência?

Além disso, esta análise pode ter implicações mais amplas, incluindo a identificação de medidas de segurança mais eficazes em viagens marítimas e a compreensão dos comportamentos humanos em situações de crise. Portanto, o problema vai além da mera previsão de sobrevivência; ele busca fornecer insights valiosos que podem informar políticas e práticas de segurança em contextos similares.

> **Links Úteis**:
> - [Objetivos, Problema de pesquisa e Justificativa](https://medium.com/@versioparole/objetivos-problema-de-pesquisa-e-justificativa-c98c8233b9c3)
> - [Matriz Certezas, Suposições e Dúvidas](https://medium.com/educa%C3%A7%C3%A3o-fora-da-caixa/matriz-certezas-suposi%C3%A7%C3%B5es-e-d%C3%BAvidas-fa2263633655)
> - [Brainstorming](https://www.euax.com.br/2018/09/brainstorming/)

## Questão de pesquisa

Qual a influência dos diferentes atributos dos passageiros, como idade, parentes a bordo sexo e classe socioeconômica, em suas chances de sobrevivência no naufrágio do Titanic?

> **Links Úteis**:
> - [Questão de pesquisa](https://www.enago.com.br/academy/how-to-develop-good-research-question-types-examples/)
> - [Problema de pesquisa](https://blog.even3.com.br/problema-de-pesquisa/)

## Objetivos preliminares

Objetivo Geral: 
Experimentar modelos de aprendizado de máquina adequados para prever a sobrevivência dos passageiros do Titanic.

Objetivos Específicos:
Avaliar o impacto da idade e sexo na probabilidade de sobrevivência.
Analisar a relação entre classe socioeconômica e chances de sobrevivência.
 
> **Links Úteis**:
> - [Objetivo geral e objetivo específico: como fazer e quais verbos utilizar](https://blog.mettzer.com/diferenca-entre-objetivo-geral-e-objetivo-especifico/)

## Justificativa

A análise da sobrevivência no desastre do Titanic não apenas oferece insights históricos, mas também pode ser útil para entender padrões de comportamento humano em situações de crise. Além disso, a identificação dos principais fatores que influenciam a sobrevivência pode informar medidas de segurança em viagens marítimas e outros contextos similares. Dados reais sobre desastres como o do Titanic são raros e valiosos para estudos acadêmicos e práticos.

> **Links Úteis**:
> - [Como montar a justificativa](https://guiadamonografia.com.br/como-montar-justificativa-do-tcc/)

## Público-Alvo

Inclui pesquisadores acadêmicos interessados em estudos históricos e comportamentais, profissionais de segurança em viagens marítimas e entusiastas de aprendizado de máquina. Os usuários podem variar em seu nível de conhecimento técnico, desde iniciantes até especialistas em análise de dados.
Um possivel viajante.

> **Links Úteis**:
> - [Público-alvo](https://blog.hotmart.com/pt-br/publico-alvo/)
> - [Como definir o público alvo](https://exame.com/pme/5-dicas-essenciais-para-definir-o-publico-alvo-do-seu-negocio/)
> - [Público-alvo: o que é, tipos, como definir seu público e exemplos](https://klickpages.com.br/blog/publico-alvo-o-que-e/)
> - [Qual a diferença entre público-alvo e persona?](https://rockcontent.com/blog/diferenca-publico-alvo-e-persona/)

## Estado da arte

* Analysis of Titanic Disaster using Machine Learning Algorithms: 

https://www.researchgate.net/publication/353352089_Analysis_of_Titanic_Disaster_using_Machine_Learning_Algorithms

* Survival prediction of Titanic disaster using machine learning: Re-visit with Neural Network Approach

https://www.jetir.org/papers/JETIR2209140.pdf

* A Comprehensive Study of Classification Algorithms on Titanic Dataset

https://ieeexplore.ieee.org/abstract/document/8229835

> **Links Úteis**:
> - [Google Scholar](https://scholar.google.com/)
> - [IEEE Xplore](https://ieeexplore.ieee.org/Xplore/home.jsp)
> - [Science Direct](https://www.sciencedirect.com/)
> - [ACM Digital Library](https://dl.acm.org/)

# Descrição do _dataset_ selecionado

O Dataset  consiste em várias variáveis que descrevem os passageiros a bordo do navio, como idade, sexo, classe socioeconômica, número de irmãos/cônjuges a bordo, número de pais/filhos a bordo, tarifa paga, número do ticket, número da cabine e se sobreviveram ou não. 

Link: https://www.kaggle.com/datasets/yasserh/titanic-dataset/data

# Canvas analítico

Nesta seção, você deverá estruturar o seu Canvas Analítico. O Canvas Analítico tem o papel de registrar a organização das ideias e apresentar o modelo de negócio.

> **Links Úteis**:
> - [Modelo do Canvas Analítico](https://github.com/ICEI-PUC-Minas-PMV-SI/PesquisaExperimentacao-Template/blob/main/help/Software-Analtics-Canvas-v1.0.pdf)

# Referências

Inclua todas as referências (livros, artigos, sites, etc) utilizados no desenvolvimento do trabalho utilizando o padrão ABNT.

> **Links Úteis**:
> - [Padrão ABNT PUC Minas](https://portal.pucminas.br/biblioteca/index_padrao.php?pagina=5886)
