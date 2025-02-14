# trafficsignrecognize
Đồ án nhận diện biển báo cấm đơn trong ảnh môi trường có sử dụng deep learning.<br>

<h2>Yêu cầu:</h2>
<b>Python:</b><br>
Code train model tại thư mục 
<pre>train_model_traffic_sign_recognize </pre>


Di chuyển đến thư mục chứa project
<pre>cd parentProjectPath/traffic_sign_recognize-master </pre>
Chạy server
<pre>python manage.py runserver</pre><br>
Sau khi chạy server thành công truy cập địa chỉ <a href="http://localhost:8000/" target="_blank">localhost:8000</a> để thao tác.
<hr>
<b>Hỗ trợ các biển báo:</b> (Theo bộ biển báo chuẩn Việt Nam)<br>
<ul>
     <li>101: Đường cấm</li>
     <li>102: Cấm đi ngược chiều</li>
     <li>122: Dừng lại</li>
     <li>127: Tốc độ tối đa cho phép</li>
</ul>

