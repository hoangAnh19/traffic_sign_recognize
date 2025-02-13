import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import matplotlib.pyplot as plt

# Load model
model_path = "model/model.h5"
model = load_model(model_path)
print("✅ Model đã load thành công!")

# Kích thước ảnh (dựa theo model)
img_height, img_width = 32, 32  # Cập nhật theo model của bạn

# Nhãn class (nếu có)
labels = [f"Class {i}" for i in range(43)]  # Thay bằng danh sách thực tế

def predict_image(img_path):
    try:
        # Đọc và hiển thị ảnh
        img = cv2.imread(img_path)
        if img is None:
            print("❌ Lỗi: Không thể load ảnh. Kiểm tra đường dẫn!")
            return

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_height, img_width))

        # Hiển thị ảnh
        plt.imshow(img)
        plt.axis("off")
        plt.show()

        # Tiền xử lý ảnh
        img = np.array(img).astype('float32') / 255.0  # Chuẩn hóa về [0,1]
        img = np.expand_dims(img, axis=0)  # (1, 32, 32, 3)

        # Dự đoán
        predictions = model.predict(img)
        print(f"🟢 Kết quả dự đoán (raw): {predictions}")

        predicted_class = np.argmax(predictions)  # Lấy class có xác suất cao nhất
        confidence = np.max(predictions)  # Xác suất lớn nhất

        print(f"🔹 Dự đoán: {labels[predicted_class]} (Độ tin cậy: {confidence:.2%})")

    except Exception as e:
        print(f"❌ Lỗi khi dự đoán: {e}")

# 🖼 Test với ảnh
test_image_path = "test.jpg"  # Thay bằng ảnh thực tế
predict_image(test_image_path)
