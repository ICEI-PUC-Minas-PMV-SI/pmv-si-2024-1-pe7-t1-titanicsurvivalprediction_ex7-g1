# Implantação da solução - Etapa 4 - Grupo 1 - Titanicsurvivalprediction

# Documentação do Serviço Web de Previsão de Sobrevivência no Titanic

## Introdução

Descrevendo as etapas de implantação de um serviço web de previsão de sobrevivência no Titanic utilizando tecnologias como Microsoft Azure, Docker, Flask, entre outras. A escolha pela Microsoft Azure deve-se à sua robustez e experiência no fornecimento de serviços de nuvem. A seguir, detalhamos a configuração do ambiente, a construção do serviço web e a implementação do modelo de machine learning.

## Etapas da Implantação

### 1. Escolha da Plataforma de Nuvem

A decisão de usar a Microsoft Azure foi baseada na experiência positiva com a plataforma, que oferece uma gama completa de serviços e ferramentas para implantação e escalabilidade de aplicações. Azure facilita a configuração de ambientes de desenvolvimento e produção, além de proporcionar integração com várias ferramentas de CI/CD.

### 2. Containerização com Docker

Docker foi utilizado para containerizar a aplicação, garantindo que o ambiente de desenvolvimento seja consistente com o de produção. Isso simplifica a implantação e a manutenção do serviço.

- **Criação do Dockerfile**: Definição das instruções de construção da imagem Docker, incluindo a instalação de dependências e a cópia do código-fonte.
- **Construção e Teste da Imagem Docker**: Construção da imagem e execução de testes para garantir que a aplicação funcione corretamente dentro do contêiner.
- **Publicação no Docker Hub**: Upload da imagem construída para um repositório no Docker Hub para facilitar a distribuição.

### 3. Desenvolvimento do Serviço Web com Flask

Flask, um microframework para Python, foi escolhido pela sua simplicidade e flexibilidade para construir APIs.

- **Configuração do Ambiente**: Instalação de Flask e outras dependências necessárias.
- **Implementação da API**: Desenvolvimento das rotas e lógica para receber dados de entrada, processá-los e retornar a previsão.

### 4. Integração com o Modelo de Machine Learning

Utilização de um modelo de RandomForest previamente treinado para realizar as previsões.

- **Carregamento do Modelo**: Utilização do `joblib` para carregar o modelo treinado.
- **Processamento dos Dados de Entrada**: Conversão dos dados JSON recebidos para um DataFrame do pandas, que é o formato esperado pelo modelo.
- **Realização da Previsão**: Uso do modelo carregado para prever se uma pessoa sobreviveu ao naufrágio do Titanic.

### 5. Configuração de CORS

Para permitir requisições de diferentes origens, foi configurado o CORS (Cross-Origin Resource Sharing) usando a biblioteca `flask_cors`.

### 6. Implantação no Microsoft Azure

- **Configuração do Serviço App no Azure**: Criação e configuração de um serviço de App no Azure para hospedar a aplicação Flask.
- **Pipeline de CI/CD**: Configuração de pipelines de integração contínua e entrega contínua para automatizar o processo de build e deployment usando Azure DevOps.

### 7. Monitoramento e Manutenção

Implementação de monitoramento e logs para garantir que o serviço web esteja funcionando corretamente e para facilitar a manutenção e solução de problemas.

### Sumário da introdução

Neste documento, detalhamos as etapas necessárias para a implantação de um serviço web de previsão de sobrevivência no Titanic, desde a escolha da plataforma de nuvem até o monitoramento da aplicação em produção. Utilizando Docker para containerização, Flask para desenvolvimento da API e Microsoft Azure para hospedagem, conseguimos construir e implantar uma solução robusta e escalável.

# Detalhamento Técnico

## Descrição do Código

O código é um exemplo de implementação de um serviço web usando Flask, um microframework para Python. Este serviço utiliza um modelo de machine learning pré-treinado (RandomForest) para prever se uma pessoa sobreviveu ao naufrágio do Titanic com base em suas características. A implementação também faz uso do Flask-CORS para permitir requisições de diferentes origens.

## Explicação do Código

### Imports

O código importa as bibliotecas necessárias.

```python
from flask import Flask, request, jsonify
import pandas as pd
import joblib
from flask_cors import CORS
```

- **Flask**: O framework web usado para criar a API.
- **request** e **jsonify**: Funções do Flask usadas para lidar com requisições HTTP e respostas JSON.
- **pandas**: Biblioteca usada para manipulação de dados, especialmente para converter dados de entrada em um DataFrame.
- **joblib**: Usado para carregar o modelo de machine learning previamente treinado.
- **CORS**: Função da biblioteca `flask_cors` que permite requisições de origens diferentes (Cross-Origin Resource Sharing).

### Inicialização do Aplicativo Flask e Configuração de CORS

```python
app = Flask(__name__)
CORS(app)
```

### Carregar o Modelo Treinado

O modelo de machine learning (RandomForest) previamente treinado é carregado a partir de um arquivo pickle (`model_pipeline.pkl`). Esse modelo inclui o pipeline completo de pré-processamento e previsão.

```python
pipeline = joblib.load('model/model_pipeline.pkl')
```

### Rota de Previsão

Define uma rota `/predict` que aceita requisições HTTP POST.

```python
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = pd.DataFrame([data])
```

- `data = request.json`: Extrai os dados JSON do corpo da requisição.
- `input_data = pd.DataFrame([data])`: Converte os dados JSON em um DataFrame do pandas, que é o formato esperado pelo modelo de machine learning.

### Realizar a Previsão com o Modelo RandomForest

Usa o modelo carregado para realizar previsões.

```python
    pred = pipeline.predict(input_data)
    pred_proba = pipeline.predict_proba(input_data)[:, 1]
```

- `pred = pipeline.predict(input_data)`: Faz a previsão (0 ou 1) para a entrada fornecida.
- `pred_proba = pipeline.predict_proba(input_data)[:, 1]`: Calcula a probabilidade associada à classe positiva (sobreviver).

### Tradução da Previsão

Converte a previsão binária (0 ou 1) em um texto compreensível ("Sobreviveu" ou "Não Sobreviveu").

```python
    prediction_text = "Sobreviveu" if pred[0] == 1 else "Não Sobreviveu"
```

### Resposta JSON

Cria uma resposta JSON que inclui a previsão textual e a probabilidade de sobrevivência.

```python
    return jsonify({
        'Prediction': prediction_text,
        'Probability': float(pred_proba[0])
    })
```

- `jsonify(...)`: Converte o dicionário Python em uma resposta JSON.

### Execução do Servidor

Inicia o servidor Flask.

```python
if __name__ == '__main__':
    app.run(debug=True)
```

- `if __name__ == '__main__':`: Verifica se o script está sendo executado diretamente.
- `app.run(debug=True)`: Inicia o servidor Flask em modo de debug.

## Código Completo

```python
from flask import Flask, request, jsonify
import pandas as pd
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Carregar o pipeline treinado
pipeline = joblib.load('model/model_pipeline.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = pd.DataFrame([data])

    # Realizar a previsão com o modelo RandomForest
    pred = pipeline.predict(input_data)
    pred_proba = pipeline.predict_proba(input_data)[:, 1]

    # Traduzir a previsão para "Sobreviveu" ou "Não Sobreviveu"
    prediction_text = "Sobreviveu" if pred[0] == 1 else "Não Sobreviveu"

    return jsonify({
        'Prediction': prediction_text,
        'Probability': float(pred_proba[0])
    })

if __name__ == '__main__':
    app.run(debug=True)
```

## Monitoramento e Manutenção

Para garantir que o serviço web de previsão de sobrevivência no Titanic esteja sempre funcionando corretamente e para facilitar a manutenção, foram implementadas várias práticas e ferramentas de monitoramento e manutenção. Essas práticas são essenciais para identificar e resolver problemas rapidamente, além de garantir a continuidade do serviço. 

Segue nosso Dashboard das Métricas selecionadas para monitoramento

![](/docs/img/monitoramento_titanicsp.png)

## Sumário

Este código cria um serviço web que usa um modelo de machine learning para prever se uma pessoa sobreviveu ao naufrágio do Titanic com base em suas características. Ele recebe dados em formato JSON via requisições HTTP POST, faz a previsão usando um modelo RandomForest carregado e retorna a previsão junto com a probabilidade associada em formato JSON.

# Apresentação da solução

Nesta seção, um vídeo de, no máximo, 5 minutos onde deverá ser descrito o escopo todo do projeto, um resumo do trabalho desenvolvido, incluindo a comprovação de que a implantação foi realizada e, as conclusões alcançadas.

