from flask import Flask, render_template, request, jsonify
import torch
import joblib
import numpy as np

app = Flask(__name__)

# 초기 모델 로드 (없으면 새로 훈련 시작)
model = SimpleNN()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    data = request.get_json()
    X_train = np.array(data['features'])
    y_train = np.array(data['labels'])
    
    # 모델 훈련
    global model
    model = train_model(X_train, y_train)
    
    # 훈련 완료 후 모델 저장
    joblib.dump(model, 'model.pth')

    return jsonify({'status': 'training completed'})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features'])
    
    # 예측
    model.eval()  # 평가 모드 활성화
    features_tensor = torch.tensor(features, dtype=torch.float32)
    prediction = model(features_tensor).detach().numpy()

    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
