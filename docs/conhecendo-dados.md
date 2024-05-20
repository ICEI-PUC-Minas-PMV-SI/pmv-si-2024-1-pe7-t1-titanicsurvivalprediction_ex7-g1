# Conhecendo os dados
[Link do Projeto 1ª parte, Conhecendo os dados](/src/notebook-g1-titanicsp.md)

A análise exploratória do conjunto de dados "Titanic-Dataset" revelou insights importantes sobre os fatores que influenciaram a sobrevivência dos passageiros durante o trágico naufrágio do RMS Titanic. O gráfico de distribuição da sobrevivência por sexo destacou
uma grande diferença  entre homens e mulheres, com a maioria das mulheres sendo resgatadas, enquanto a maioria dos homens não sobreviveu. Essa constatação reflete a política de "mulheres e crianças primeiro" adotada nos esforços de salvamento. Além disso, o histograma
da sobrevivência por idade mostrou que as crianças e passageiros mais jovens, apresentaram uma taxa de sobrevivência significativamente maior do que os adultos, sendo também priorizados durante o resgate. Esses dados evidenciam como o gênero e a idade foram fatores 
cruciais na probabilidade de sobrevivência, com mulheres e crianças recebendo prioridade nos botes salva-vidas.

Outro aspecto importante foi o papel da classe social dos passageiros. A distribuição da sobrevivência por classe revelou que aqueles pertencentes às classes superiores (primeira e segunda classe) tiveram muito mais chances de serem resgatados do que os passageiros da 
terceira classe.Isso ressalta o impacto da desigualdade socioeconômica da época, mostrando como o status e os recursos financeiros afetaram diretamente o acesso aos meios de sobrevivência.Os dados também sugeriram que a presença de dependentes, como cônjuges ou filhos, 
pode ter aumentado as chances de sobrevivência dos passageiros. Aqueles com dependentes parecem ter sido priorizados nos esforços de resgate, provavelmente devido à necessidade de preservar as famílias durante a evacuação.

As medidas de dispersão, como o desvio padrão de 14,5 anos para idade e 49,7 para as tarifas, juntamente com o histograma de idade e o box plot de tarifas, indicaram uma variabilidade considerável nesses aspectos. Enquanto a média de idade era de 29,7 anos, a distribuição 
dos dados mostrou concentrações em faixas etárias específicas, além de valores atípicos. Da mesma forma, embora a média de sobrevivência tenha sido de 38,4%, a ampla variação nas tarifas pagas pelos passageiros mostrou diferenças substanciais nos recursos financeiros disponíveis.

**Detalhamento** 

> ### Distribuição da Sobrevivência por Sexo:
> ---
> ![Tabela Percentual de Passageiros a Bordo por Sexo](/docs/img/tabelaSexo.png)
>
>![Gráfico Percentual de Passageiros a Bordo por Sexo](/docs/img/output_13_0.png)



> ### Distribuição da Sobrevivência por Idade:
> ---
> <div>
> <table border="1" class="dataframe">
>  <thead>
>    <tr style="text-align: right;">
>      <th></th>
>      <th>Age</th>
>    </tr>
>  </thead>
>  <tbody>
>    <tr>
>      <th>count</th>
>      <td>714.000000</td>
>    </tr>
>    <tr>
>      <th>mean</th>
>      <td>29.699118</td>
>    </tr>
>    <tr>
>      <th>std</th>
>      <td>14.526497</td>
>    </tr>
>    <tr>
>      <th>min</th>
>      <td>0.420000</td>
>    </tr>
>    <tr>
>      <th>25%</th>
>      <td>20.125000</td>
>    </tr>
>    <tr>
>      <th>50%</th>
>      <td>28.000000</td>
>    </tr>
>    <tr>
>      <th>75%</th>
>      <td>38.000000</td>
>    </tr>
>    <tr>
>      <th>max</th>
>      <td>80.000000</td>
>    </tr>
>  </tbody>
> </table>
> </div>
>
> ![](/docs/img/tabelaIdade.png)
>
> ![](/docs/img/output_16_0.png)



> ### Distribuição da Sobrevivência por Classe Social:
> ---
> ![](/docs/img/tabelaClasse.png)
>
> ![](/docs/img/output_19_0.png)



> ### Distribuição da Sobrevivência por Sobreviventes por Dependentes-Irmãos/Cônjuge:
> ---
> ![](/docs/img/tabelaDepIrmaosC.png)
>
> ![](/docs/img/output_21_0.png)



> ### Distribuição da Sobrevivência por Sobreviventes por Dependentes-Pais/Filhos
> ---
> ![](/docs/img/tabelaDepPaisFilhos.png)
>
> ![](/docs/img/output_23_0.png)



Essas análises destacaram a importância dos fatores como gênero, idade, classe social e presença de dependentes na determinação de quem sobreviveria ao desastre do Titanic. Elas revelaram as prioridades e desafios enfrentados em situações de emergência, além das desigualdades 
sociais e econômicas prevalecentes no início do século XX. Embora as medidas de tendência central forneçam uma visão geral, as medidas de dispersão e as visualizações gráficas foram  fundamentais para revelar a complexidade das relações entre as variáveis e seu impacto 
na probabilidade de sobrevivência.

Esses resultados não apenas enriquecem nossa compreensão histórica do evento, mas também destacam as consequências das disparidades sociais e a necessidade de uma resposta equitativa e coordenada em situações de emergência. A análise exploratória desse conjunto de dados
fornece insights valiosos sobre a natureza humana e as complexidades envolvidas na tomada de decisões em circunstâncias extremas.

## Descrição dos achados

A Matriz de Correlação, como o próprio nome induz mostra as correlações entre diferentes variáveis do conjunto de dados do naufrágio do Titanic. Cada célula colorida representa o coeficiente de correlação entre duas variáveis específicas. As cores mais avermelhadas indicam correlação positiva, enquanto as cores mais alaranjadas/amareladas indicam correlação negativa.
 
Algumas observações importantes:
 
1.⁠ ⁠A variável "PassengerId" tem correlação próxima de 1 consigo mesma, o que é esperado, já que é um identificador único.
 
2.⁠ ⁠As variáveis "Survived" e "Pclass" mostram uma correlação negativa moderada, sugerindo que passageiros de classes mais altas tiveram maior probabilidade de sobrevivência.
 
3.⁠ ⁠"Age" mostra correlações negativas fracas com "Survived" e "Pclass", indicando que pessoas mais jovens tinham uma leve vantagem de sobrevivência e tendiam a estar em classes mais altas.
 
4.⁠ ⁠"Sibsp" (número de irmãos/cônjuges a bordo) tem uma correlação positiva fraca com "Parch" (número de pais/filhos a bordo), sugerindo que famílias maiores viajavam juntas.
 
5.⁠ ⁠"Fare" tem uma correlação positiva moderada com "Pclass", o que faz sentido, já que passageiros de classes mais altas pagavam tarifas mais altas.
 
Em resumo, este gráfico de correlações pode fornecer insights iniciais sobre os relacionamentos entre as variáveis do conjunto de dados do Titanic e orientar uma análise mais aprofundada. Por exemplo, seria possível presumir qual seria o percentual de uma pessoa sobrevir a um naufrágio semelhante ao do Titanic se fosse lavado em consideração sua classe social, idade e sexo, caso fossem utilizados critérios semelhantes para categorizar a prioridade de embarque em um bote salva-vidas. 

## Ferramentas utilizadas

**Notebooks Kaggle:** Utilizamos o Kaggle, pois oferece uma interface de notebook usando o Jupyter Notebook, que é integrada e suporta kernels para Python. Usamos esses notebooks para escrever e executar nosso código de análise de dados de forma colaborativa. A integração com o ambiente Kaggle facilitou o compartilhamento e a colaboração com outros membros da comunidade.

**Bibliotecas Python:** Na plataforma Kaggle, encontramos suporte para uma ampla gama de bibliotecas populares de Python, como Pandas, NumPy, Matplotlib, Seaborn, Tabulate entre outras. Essas bibliotecas foram utilizadas para realizar manipulação de dados, visualização de gráficos, tabelas e análise estatística.


