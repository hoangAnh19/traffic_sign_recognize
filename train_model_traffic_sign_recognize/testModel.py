import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import matplotlib.pyplot as plt

# Load model
model_path = "./model/model.h5"
model = load_model(model_path)
print("✅ Model đã load thành công!")
print(f"✅ Model input shape: {model.input_shape}")

# Kích thước ảnh (dựa theo model)
img_height, img_width = 30, 30  # Cập nhật theo model của bạn

# Nhãn class (nếu có)
labels = [f"Class {i}" for i in range(43)]  # Thay bằng danh sách thực tế


def predict_image(img_path):
    try:
        # Đọc ảnh
        img = cv2.imread(img_path)
        if img is None:
            print("❌ Lỗi: Không thể load ảnh. Kiểm tra đường dẫn!")
            return

        print(f"🖼 Ảnh gốc có shape: {img.shape}")  # Debug kích thước ảnh

        # Chuyển ảnh sang RGB nếu cần
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize ảnh về kích thước mong muốn
        img = cv2.resize(img, (img_height, img_width))

        print(f"📏 Ảnh sau resize có shape: {img.shape}")  # Debug kích thước ảnh

        # Hiển thị ảnh
        # plt.imshow(img)
        # plt.axis("off")
        # plt.show()

        # Kiểm tra số kênh màu
        if len(img.shape) == 2:  # Nếu ảnh chỉ có 1 kênh (grayscale)
            print("⚠️ Ảnh đang ở dạng grayscale, cần chuyển về 3 kênh màu!")
            img = np.stack([img] * 3, axis=-1)  # Chuyển thành (32, 32, 3)

        print(f"🎨 Ảnh đầu vào có shape (sau khi xử lý màu): {img.shape}")

        # Tiền xử lý ảnh
        img = np.array(img).astype('float32') / 255.0  # Chuẩn hóa về [0,1]
        img = np.expand_dims(img, axis=0)  # (1, 32, 32, 3)

        print(f"🚀 Shape cuối cùng trước khi đưa vào model: {img.shape}")

        # Dự đoán
        predictions = model.predict(img)
        print(f"🟢 Kết quả dự đoán (raw): {predictions}")

        predicted_class = np.argmax(predictions)  # Lấy class có xác suất cao nhất
        confidence = np.max(predictions)  # Xác suất lớn nhất

        print(f"🔹 Dự đoán: Class {predicted_class} (Độ tin cậy: {confidence:.2%})")
        print(predicted_class)

    except Exception as e:
        print(f"❌ Lỗi khi dự đoán: {e}")


# 🖼 Test với ảnh
test_image_path = "original.jpeg"  # Thay bằng đường dẫn ảnh thực tế
predict_image(test_image_path)