from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.parse
import json
import base64
from PIL import Image
import io
import cv2
import numpy as np

def base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    return image_data

def base64_to_opencvimage(base64_string):
    # 使用base64库解码Base64编码的字符串
    decoded_data = base64.b64decode(base64_string)
    
    # 使用numpy将解码的字节串转换为numpy数组
    np_array = np.frombuffer(decoded_data, np.uint8)
    
    # 使用OpenCV从numpy数组创建一个图像矩阵
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    return img

def save_image(image_data, file_path):
    image = Image.open(io.BytesIO(image_data))
    image.save(file_path)


class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

    def do_POST(self):
        # 获取POST请求的内容长度
        content_length = int(self.headers['Content-Length'])
        # 读取POST请求的内容
        post_data = self.rfile.read(content_length)
        # print("type: ", type(post_data))
        # print("post: ", post_data)

        json_data = json.loads(post_data)
        print(f"json_data: {json_data}")

        image_base64 = json_data["image_base64"]
        img = base64_to_opencvimage(image_base64)

        cv2.imwrite(f"save_test_img.jpg", img)


        json_data["image_base64"] = json_data["image_base64"][0:100]


        
        # 解析POST请求的数据
        # parsed_data = urllib.parse.parse_qs(post_data.decode('utf-8'))
        
        # 响应客户端
        self.send_response(200)  # 发送状态码200
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        
        # 发送响应内容
        # response_message = f"Received POST request: {parsed_data}"
        response_message = f"Received POST request: {json_data}"
        self.wfile.write(response_message.encode('utf-8'))

# 设置服务器端口
port = 8081
server_address = ('', port)

# 创建HTTP服务器
httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)

# 开始运行服务器
print(f"Server running on port {port}...")
httpd.serve_forever()