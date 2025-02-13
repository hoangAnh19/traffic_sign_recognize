import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model

# 🔹 Load model đã huấn luyện
try:
    model = load_model("traffic_sign_model.h5")
    print("✅ Model đã load thành công!")
except Exception as e:
    print(f"❌ Lỗi khi load model: {e}")
    exit()


# 🔹 Hàm tiền xử lý ảnh
def preprocess_image(image_path, input_size=(64, 64)):
    """
    Đọc và xử lý ảnh trước khi đưa vào mô hình để dự đoán.
    """
    print(f"🔍 Đang kiểm tra ảnh: {image_path}")

    # Kiểm tra đường dẫn ảnh
    if not os.path.exists(image_path):
        raise ValueError(f"❌ Ảnh không tồn tại: {image_path}")

    # Đọc ảnh
    img = cv2.imread(image_path)

    # Kiểm tra ảnh có tồn tại không
    if img is None:
        raise ValueError(f"❌ Không thể đọc ảnh: {image_path}")

    # Chuyển sang RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize về kích thước chuẩn của mô hình
    img_resized = cv2.resize(img, input_size)

    # Chuẩn hóa giá trị pixel về [0,1]
    img_resized = img_resized.astype("float32") / 255.0

    # Thêm batch dimension (1, 64, 64, 3)
    img_resized = np.expand_dims(img_resized, axis=0)

    return img, img_resized  # Trả về cả ảnh gốc và ảnh đã resize


# 🔹 Hàm dự đoán biển báo từ ảnh
def predict_traffic_sign(image_path, class_labels):
    """
    Dự đoán biển báo giao thông từ ảnh đầu vào và hiển thị kết quả trên ảnh.
    """
    try:
        # Tiền xử lý ảnh
        original_img, input_image = preprocess_image(image_path)

        # Dự đoán với mô hình
        predictions = model.predict(input_image)

        # Kiểm tra output của model
        print(f"📊 Dự đoán raw: {predictions}")

        # Lấy nhãn có xác suất cao nhất
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions)  # Lấy xác suất cao nhất
        predicted_label = class_labels[predicted_class]

        # Hiển thị kết quả lên ảnh
        img_with_text = display_prediction(original_img, predicted_label, confidence)

        # Hiển thị ảnh có kết quả dự đoán
        plt.imshow(img_with_text)
        plt.title(f"{predicted_label} ({confidence:.2%})")
        plt.axis("off")
        plt.show()

        return predicted_class

    except Exception as e:
        print(f"❌ Lỗi khi dự đoán: {e}")
        return None


# 🔹 Hàm hiển thị dự đoán lên ảnh
def display_prediction(img, label, confidence):
    """
    Vẽ nhãn và độ chính xác lên ảnh.
    """
    # Chuyển ảnh về BGR để dùng OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Thiết lập font
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (0, 255, 0)  # Màu xanh lá

    # Vị trí text trên ảnh
    text = f"{label} ({confidence:.2%})"
    position = (30, 50)

    # Vẽ chữ lên ảnh
    # cv2.putText(img, text, position, font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    #
    # # Chuyển ảnh về RGB trước khi hiển thị bằng matplotlib
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# 🔹 Danh sách nhãn biển báo (cập nhật theo dataset GTSRB)
class_labels = [
    "Biển báo 20km/h", "Biển báo 30km/h", "Biển báo 50km/h", "Biển báo 60km/h",
    "Biển báo 70km/h", "Biển báo 80km/h", "Hết hạn chế tốc độ", "Biển báo 100km/h",
    "Biển báo 120km/h", "Cấm vượt", "Cấm xe tải vượt", "Đường ưu tiên",
    "Nhường đường", "Dừng lại", "Cấm xe", "Cấm xe tải", "Cấm vào", "Nguy hiểm",
    "Khúc cua trái", "Khúc cua phải", "Đường vòng", "Đường gập ghềnh",
    "Đường hẹp phải", "Đường hẹp trái", "Công trường", "Đèn giao thông",
    "Người đi bộ", "Trẻ em", "Đi xe đạp", "Đường trơn", "Cảnh báo đường hẹp",
    "Cảnh báo động vật", "Hết giới hạn tốc độ", "Rẽ phải", "Rẽ trái",
    "Đi thẳng", "Đi thẳng hoặc rẽ phải", "Đi thẳng hoặc rẽ trái",
    "Đi bên phải", "Đi bên trái", "Đường một chiều", "Dừng xe",
    "Hết hạn chế giao thông", "Cấm dừng xe"
]

# 🔹 Kiểm tra số lượng lớp output của mô hình
num_classes = model.output_shape[-1]
print(f"🔢 Model có {num_classes} lớp output")

if len(class_labels) != num_classes:
    print(f"⚠️ Cảnh báo: Model có {num_classes} class nhưng bạn có {len(class_labels)} nhãn!")
    print("⚠️ Hãy kiểm tra lại file model hoặc danh sách nhãn!")

# 🔹 Đường dẫn ảnh test (THAY BẰNG ĐƯỜNG DẪN ẢNH THỰC TẾ)
image_path = "test_image2.jpg"

# 🔹 Dự đoán biển báo giao thông
predict_traffic_sign(image_path, class_labels)
