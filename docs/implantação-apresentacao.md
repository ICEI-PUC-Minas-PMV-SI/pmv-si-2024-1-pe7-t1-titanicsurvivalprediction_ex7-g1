# Implantação da solução

# Documentação do Serviço Web de Previsão de Sobrevivência no Titanic

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

## Sumário

Este código cria um serviço web que usa um modelo de machine learning para prever se uma pessoa sobreviveu ao naufrágio do Titanic com base em suas características. Ele recebe dados em formato JSON via requisições HTTP POST, faz a previsão usando um modelo RandomForest carregado e retorna a previsão junto com a probabilidade associada em formato JSON.

# Apresentação da solução

Nesta seção, um vídeo de, no máximo, 5 minutos onde deverá ser descrito o escopo todo do projeto, um resumo do trabalho desenvolvido, incluindo a comprovação de que a implantação foi realizada e, as conclusões alcançadas.

