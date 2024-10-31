from http.server import BaseHTTPRequestHandler, HTTPServer
import http.server
import urllib.parse
import json
import base64
from PIL import Image
import io
import cv2
import numpy as np
from ultralytics import YOLO

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

model_det_helmet = YOLO("helmet_241009.pt")
model_cls_working_clothes = YOLO("cls_working_clothes_best.pt")

def infer_check(img):
    '''返回元组， 第一位表示是否有戴安全帽， 第二位表示是否有穿工装'''
    result_have_helmet = 0
    result_have_working_clothes = 0

    results = model_det_helmet.predict(source=img, save=True, save_txt=True)
    boxes = results[0].boxes

    if len(boxes.cls) == 0:
        return result_have_helmet, result_have_working_clothes
    
    for cls_id in boxes.cls:
        if cls_id == 2:
            result_have_helmet = 1
            break


    max_index = 0
    max_area = boxes.xywh[max_index][2] * boxes.xywh[max_index][3]
    for i in range(1, len(boxes.cls)):
        area = boxes.xywh[i][2] * boxes.xywh[i][3]
        if area > max_area:
            max_area = area
            max_index = i

    cls_id = boxes.cls[max_index]
    conf = boxes.conf[max_index]

    
    if cls_id == 0 or cls_id == 1: # personup or persondown
        box = boxes.xyxy[max_index]
        croped_img = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]

        cls_results = model_cls_working_clothes.predict(source=croped_img, save=True, save_txt=True)

        cls_top1 = cls_results[0].probs.top1
        cls_top1_conf = cls_results[0].probs.top1conf

        if cls_top1 == 2:                # "other_clothes", "unknow", "working_clothes"
            result_have_working_clothes = 1
    
    return result_have_helmet, result_have_working_clothes



class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

    def do_POST(self):
        if self.path == '/api/ai_check':
            # 获取POST请求的内容长度
            content_length = int(self.headers['Content-Length'])
            # 读取POST请求的内容
            post_data = self.rfile.read(content_length)
            # print("type: ", type(post_data))
            # print("post: ", post_data)

            json_data = json.loads(post_data)
            # print(f"json_data: {json_data}")

            image_base64 = json_data["image_base64"]
            img = base64_to_opencvimage(image_base64)

            result_have_helmet, result_have_working_clothes = infer_check(img)

            cv2.imwrite(f"save_test_img.jpg", img)   
        
            # json_data["image_base64"] = json_data["image_base64"][0:100]

            result_json = {"image_id": 0, 
                        "result": {
                            "helmet": 0,
                            "working_clothes": 1
                            }}
            result_json["image_id"] = json_data["image_id"]
            result_json["result"]["helmet"] = result_have_helmet
            result_json["result"]["working_clothes"] = result_have_working_clothes

            print("result_json: ", result_json)

            
            # 解析POST请求的数据
            # parsed_data = urllib.parse.parse_qs(post_data.decode('utf-8'))
            
            # 响应客户端
            self.send_response(200)  # 发送状态码200
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            
            # 发送响应内容
            # response_message = f"Received POST request: {parsed_data}"
            response_message = f"Received POST request: {result_json}"
            self.wfile.write(response_message.encode('utf-8'))

        else:
            print("error self.path: ", self.path)
            return http.server.SimpleHTTPRequestHandler.do_POST(self)




if __name__ == "__main__":
    # 设置服务器端口
    port = 9091
    server_address = ('', port)

    # 创建HTTP服务器
    httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)

    # 开始运行服务器
    print(f"Server running on port {port}...")
    httpd.serve_forever()