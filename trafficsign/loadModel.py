from tensorflow.keras.models import load_model
import os

# model_path = "static/model/model.h5"  # Đường dẫn đến model của bạn
currentPythonFilePath = os.getcwd()
# modelUrl = currentPythonFilePath+'/static/model/model.h5'
modelUrl = '../static/model/model.h5'
# D:\Source Code\traffic_sign_recognize\static\model

try:
    model = load_model(modelUrl)
    print("✅ Model đã load thành công!")
    print(model.summary())
except Exception as e:
    print(f"❌ Lỗi khi load model: {e}")


