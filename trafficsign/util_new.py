from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
from django.views.generic import TemplateView
from django.core.files.storage import FileSystemStorage
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import random
from tensorflow.keras.models import load_model
from PIL import Image
import os
import sys


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def uploadFile(request):
    context = {}
    if request.method == 'POST':
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage()
        name = fs.save(uploaded_file.name, uploaded_file)
        context['url'] = fs.url(name)
    return context


def imreadx(url):
    img = io.imread(url)
    outimg = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return outimg


def imshowx(img, title='B2DL'):
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12
    fig_size[1] = 4
    plt.rcParams["figure.figsize"] = fig_size

    plt.axis('off')
    plt.title(title)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


def imshowgrayx(img, title='BD2L'):
    plt.axis('off')
    plt.title(title)
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.show()



def cropAndDetectTrafficSign(context):
    try:
        currentPythonFilePath1 = os.getcwd()
        currentPythonFilePath2 = os.getcwd()
        url_image = context['url']
        print(f"url_image: {url_image}")
        print(f"currentPythonFilePath1: {os.getcwd()}")

        # Chỉnh sửa đường dẫn để sử dụng os.path.join thay vì + để đảm bảo tính tương thích trên nhiều hệ điều hành
        model_path = os.path.join(currentPythonFilePath1, 'trafficsign', 'model', 'model.h5')

        # Tạo đường dẫn ảnh từ url_image và thay thế dấu '/' bằng '\'
        saveDetectImageUrl11 = currentPythonFilePath2+ url_image
        print(f"saveDetectImageUrl11 path: {saveDetectImageUrl11}")
        saveDetectImageUrl = saveDetectImageUrl11.replace('/', '\\')  # Chuyển đường dẫn sang định dạng Windows
        print(f"saveDetectImageUrl path: {saveDetectImageUrl}")
        print(f"Model path: {model_path}")


        # Kiểm tra file ảnh tồn tại trước khi load
        # if not os.path.exists(saveDetectImageUrl):
        #     print(f"❌ Lỗi: Không tìm thấy ảnh tại đường dẫn {saveDetectImageUrl}")
        #     return

        # Tải model
        model = load_model(model_path)

        # Đọc ảnh từ đường dẫn đã xác định
        img = cv2.imread(saveDetectImageUrl)

        if img is None:
            print("❌ Lỗi: Không thể load ảnh. Kiểm tra đường dẫn!")
            print(f"❌ Lỗi: Không tìm thấy ảnh tại đường dẫn {saveDetectImageUrl}")
            return

        print(f"🖼 Ảnh gốc có shape: {img.shape}")  # Debug kích thước ảnh

        # Chuyển ảnh sang RGB nếu cần
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize ảnh về kích thước mong muốn
        img_height, img_width = 30, 30
        img = cv2.resize(img, (img_height, img_width))

        print(f"📏 Ảnh sau resize có shape: {img.shape}")  # Debug kích thước ảnh

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

        return predicted_class

    except Exception as e:
        print(f"❌ Lỗi khi dự đoán: {e}")


def detectTrafficSign(request):
    context = uploadFile(request)
    prediction = cropAndDetectTrafficSign(context)
    print(f"🟢 Kết quả dự đoán : {prediction}")
    context['traffictrainid'] = prediction
    return context
