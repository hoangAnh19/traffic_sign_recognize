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
        currentPythonFilePath = os.getcwd()
        modelUrl = os.path.join(currentPythonFilePath, 'trafficsign', 'model', 'model.h5')

        # saveDetectImageUrl = currentPythonFilePath + context['url']
        url = currentPythonFilePath + context['url']
        saveDetectImageUrl = currentPythonFilePath+'/static/image/'# Chuyển đường dẫn sang định dạng Windows
        imageType = url.split('.')[1]
        print(f"url path: {url}")

        img = cv2.imread(url)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_r1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
        mask_r2 = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
        mask_r = cv2.bitwise_or(mask_r1, mask_r2)
        target = cv2.bitwise_and(img, img, mask=mask_r)
        gblur = cv2.GaussianBlur(mask_r, (9, 9), 0)
        edge_img = cv2.Canny(gblur, 30, 150)

        cv2.imwrite(saveDetectImageUrl + 'original.' + imageType, img)
        cv2.imwrite(saveDetectImageUrl + 'markrange1.' + imageType, mask_r1)
        cv2.imwrite(saveDetectImageUrl + 'markrange2.' + imageType, mask_r2)
        cv2.imwrite(saveDetectImageUrl + 'maskforredregion.' + imageType, mask_r)
        cv2.imwrite(saveDetectImageUrl + 'maskforredrigon.' + imageType, target)
        cv2.imwrite(saveDetectImageUrl + 'edgemap.' + imageType, edge_img)

        img2 = img.copy()

        # Điều chỉnh ở đây, nếu sử dụng OpenCV 4.x, bạn chỉ cần giải nén 2 giá trị
        cnts, hierarchy = cv2.findContours(edge_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(img2, cnts, -1, (0, 255, 0), 2)
        cv2.imwrite(saveDetectImageUrl + 'contournorestriction.' + imageType, img2)

        img2 = img.copy()
        try:
            for cnt in cnts:
                area = cv2.contourArea(cnt)
                if area < 1000:
                    continue
                ellipse = cv2.fitEllipse(cnt)
                cv2.ellipse(img2, ellipse, (0, 255, 0), 2)
                x, y, w, h = cv2.boundingRect(cnt)
                a, b, c, d = x, y, w, h
                cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 3)

            cv2.imwrite(saveDetectImageUrl + 'contourrestrictedforlargeregion.' + imageType, img2)

            crop = img2[b:b + d, a:a + c]
            cv2.imwrite(saveDetectImageUrl + 'cropimage.' + imageType, crop)

            # Tải model
            model = load_model(modelUrl)

            # Tiền xử lý ảnh
            data = []
            image_from_array = Image.fromarray(crop, 'RGB')
            crop = image_from_array.resize((30, 30))
            data.append(np.array(crop))

            X_test = np.array(data)
            X_test = X_test.astype('float32') / 255  # Chuẩn hóa ảnh

            # Dự đoán
            predictions = model.predict(X_test)  # Dùng model.predict thay cho predict_classes
            predicted_class = np.argmax(predictions)  # Lấy lớp có xác suất cao nhất

            return predicted_class
        except Exception as e:
            print("cannot border box")
            os.remove(saveDetectImageUrl + 'cropimage.' + imageType)
            cv2.imwrite(saveDetectImageUrl + 'cropimage.' + imageType, img)
            model = load_model(modelUrl)
            data = []
            image_from_array = Image.fromarray(img, 'RGB')
            img = image_from_array.resize((30, 30))
            data.append(np.array(img))
            X_test = np.array(data)
            X_test = X_test.astype('float32') / 255
            predictions = model.predict(X_test)  # Dự đoán lại
            predicted_class = np.argmax(predictions)  # Lấy lớp có xác suất cao nhất
            print(f"❌ Lỗi khi dự đoán1: {e}")
            return predicted_class
    except Exception as e:
        print(f"❌ Lỗi khi dự đoán2: {e}")
        return [10000]


def detectTrafficSign(request):
    context = uploadFile(request)
    prediction = cropAndDetectTrafficSign(context)
    print(f"🟢 Kết quả dự đoán : {prediction}")
    if prediction == 0:
        prediction = 1
    if prediction == 14:
        prediction = 12
    context['traffictrainid'] = prediction
    return context
