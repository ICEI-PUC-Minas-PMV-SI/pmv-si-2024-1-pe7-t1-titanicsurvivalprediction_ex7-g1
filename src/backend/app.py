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
