Descrição textual detalhada de todas as tarefas realizadas

Pré-processamento de Dados:

Carregamento do Dataset: Os dados do Titanic foram carregados a partir de um arquivo CSV.
Divisão de Recursos e Alvo: As variáveis independentes (features) e a variável dependente (target) foram separadas.
Tratamento de Valores Faltantes: Utilizamos o SimpleImputer para preencher valores faltantes nas features numéricas e categóricas.
Escalonamento: Os dados numéricos foram normalizados usando o StandardScaler para garantir que todas as features tivessem a mesma escala.
Codificação de Categorias: As variáveis categóricas foram transformadas em variáveis binárias usando o OneHotEncoder.

Treinamento dos Modelos:

Random Forest Classifier: Ajustamos o modelo com diferentes hiperparâmetros, como critério de divisão e profundidade máxima, para encontrar a configuração ideal.
Regressão Logística: Ajustamos o modelo com regularização e número máximo de iterações para otimizar o desempenho.

Avaliação do Desempenho dos Modelos

Divisão de Dados: Os dados foram divididos em conjuntos de treino e teste para avaliar o desempenho dos modelos.

Treinamento e Previsão: Os modelos foram treinados com os dados de treino e usados para fazer previsões nos dados de teste.

Métricas de Desempenho: Calculamos várias métricas, incluindo acurácia, precisão, recall, F1-Score, ROC AUC, matriz de confusão e relatório de classificação, para avaliar o desempenho de cada modelo.
Comparação dos Resultados e Seleção do Melhor Modelo:

Analisamos as métricas de desempenho de cada modelo para determinar qual teve o melhor desempenho na tarefa de classificação de sobreviventes do Titanic.
Com base na análise das métricas, identificamos o modelo mais eficaz e justificamos nossa escolha com base nos resultados obtidos.

Essas tarefas foram realizadas para construir e avaliar os modelos de aprendizado de máquina e garantir que fossem capazes de fazer previsões precisas sobre a sobrevivência dos passageiros do Titanic.
