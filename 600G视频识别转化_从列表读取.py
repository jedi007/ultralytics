# ssh root@203.25.213.139    pw:8K9mRMERPpubwM8z   /data/zlm/data/www/record

import json
from copy import deepcopy
import re
import os
 
current_path = os.path.dirname(os.path.realpath(__file__))

print("current_path: ", current_path)

videos_source_dir = "/data/cache/live0"

out_dir = "/data/cache/live_add_AI_from_list"

def file_path_exists(file_path):
    if os.path.exists(file_path):
        print(f"{file_path} 存在！")
    else:
        print(f"{file_path} 不存在！")
        b = os.mkdir(file_path)
        print(f"创建文件夹{file_path} b:{b}")

def traverse_folder(folder_path):
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            abs_file_path = os.path.join(root, file_name)
            # print(abs_file_path)
            file_list.append(abs_file_path)
    
    return file_list

def get_out_file_name(video_path):
    if not video_path.endswith(".mp4"):
        print(f"{video_path} not end with mp4")
        return ""
    
    tmp = video_path.replace(f"{videos_source_dir}/", "").split("/")

    print("tmp: ", tmp)

    # check
    if tmp[2].split(".")[1] != "mp4":
        print("get error path: ", video_path)
        return ""

    hash_code = tmp[0]
    date_str = tmp[1]
    file_name = tmp[2].split(".")[0]

    hash_path = os.path.join(out_dir, hash_code)
    file_path_exists(hash_path)

    date_path = os.path.join(hash_path, date_str)
    file_path_exists(date_path)

    return f"{date_path}/{file_name}_AI.mp4"


import cv2
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np

yolo = YOLO('det_personup_helmet_250107.pt')
def yolo_det_and_save_video(video_path, out_file_name):
    #yolo = YOLO('det_personup_helmet_250107.pt')
    #video_path = '/home/hyzh/lijie/cache/2024-03-14/10-30-42.mp4'
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    print("fps: ", fps)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter('/home/hyzh/lijie/cache/2024-03-14/10-30-42_AI.mp4', fourcc, fps, (frame_width, frame_height))
    out = cv2.VideoWriter(out_file_name, fourcc, fps, (frame_width, frame_height))

    # 加载支持中文的字体，调大字体大小，这里设置为 30
    font_path = 'font.ttf'
    font_size = 25
    font = ImageFont.truetype(font_path, font_size)

    # 定义类别名称和颜色映射
    class_names = {0: '工作服', 1: '跌倒', 2: '安全帽', 3: '未戴安全帽', 4: '安全绳', 5: '未戴安全绳'}
    color_map = {
        0: (255, 0, 0),
        1: (0, 255, 0),
        2: (0, 0, 255),
        3: (255, 255, 0),
        4: (255, 0, 255),
        5: (0, 255, 255)
    }

    frame_count = 0
    while cap.isOpened():
        status, frame = cap.read()
        if not status:
            break
        result = yolo.predict(source=frame, save=False, verbose=False)
        result = result[0]

        # 将 OpenCV 的 BGR 图像转换为 PIL 的 RGB 图像
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # 遍历检测结果，绘制中文标签和目标框
        for box in result.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            label = f"{class_names[class_id]} {conf:.2f}"
            color = color_map[class_id]

            # 绘制中文标签
            draw.text((x1, y1 - font_size - 5), label, font=font, fill=color)

            # 绘制目标框
            draw.rectangle((x1, y1, x2, y2), outline=color, width=2)

        # 将 PIL 图像转换回 OpenCV 的 BGR 图像
        anno_frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        out.write(anno_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        frame_count += 1
        if frame_count % 800 == 0:
            print("frame_count: ", frame_count)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print('保存完成')


if __name__ == '__main__':
    file_path_exists(out_dir)


    files_list = ["8146c95d-1292-4163-8277-791617f90cdf#0/2025-02-13/10-13-55.mp4",
"8146c95d-1292-4163-8277-791617f90cdf#0/2025-02-13/11-05-41.mp4",
"8146c95d-1292-4163-8277-791617f90cdf#0/2025-02-13/15-02-02.mp4",
"8146c95d-1292-4163-8277-791617f90cdf#0/2025-02-13/15-06-10.mp4",
"a2664297-08a1-420a-8fb0-c7a8e7c16f17#3/2025-02-12/16-15-17.mp4",
"062ce920-0542-4b3b-9c5f-9af2400950ec#0/2025-02-11/14-58-07.mp4",
"4bf41a2f-de14-470c-95c9-0d9420a826fb#7/2025-02-11/14-38-24.mp4",
"45a24436-702c-473e-bb46-70380291f698#4/2025-02-11/09-57-33.mp4",
"45a24436-702c-473e-bb46-70380291f698#4/2025-02-11/10-17-41.mp4",
"45a24436-702c-473e-bb46-70380291f698#4/2025-02-11/10-19-15.mp4",
"45a24436-702c-473e-bb46-70380291f698#4/2025-02-11/10-27-02.mp4",
"cc605ce5-c139-45ab-bb13-4f8414413598#4/2025-02-10/11-12-30.mp4",
"cc605ce5-c139-45ab-bb13-4f8414413598#4/2025-02-10/12-21-42.mp4",
"88d8d13a-76e2-4294-af76-392aab2ed875#7/2025-02-07/14-11-26.mp4",
"88d8d13a-76e2-4294-af76-392aab2ed875#7/2025-02-11/10-44-26.mp4",
"bfedb9a9-30b8-4b3b-a057-b60787aec23a#0/2025-01-22/10-29-47.mp4",
"bfedb9a9-30b8-4b3b-a057-b60787aec23a#0/2025-01-22/10-52-36.mp4",
"bfedb9a9-30b8-4b3b-a057-b60787aec23a#0/2025-01-22/10-56-33.mp4",
"6c9cb688-39a0-472d-8f68-9ecf1736288b#7/2025-01-22/09-54-26.mp4",
"6c9cb688-39a0-472d-8f68-9ecf1736288b#7/2025-01-22/09-58-28.mp4",
"6c9cb688-39a0-472d-8f68-9ecf1736288b#7/2025-01-22/11-27-40.mp4",
"c208f2b5-52a4-4361-8e18-7852cc918448#0/2025-01-22/09-48-11.mp4",
"c208f2b5-52a4-4361-8e18-7852cc918448#0/2025-01-22/09-51-18.mp4",
"30340014-54f1-43ad-a96a-23a7b8856011#7/2025-01-22/09-29-15.mp4",
"30340014-54f1-43ad-a96a-23a7b8856011#7/2025-01-22/14-50-22.mp4",
"5392750b-b03f-4128-99c7-920128b1c2d6#0/2025-01-20/11-01-19.mp4",
"f2bc800e-b8c4-4797-8c43-53de50bb2ec2#0/2025-02-07/10-38-01.mp4",
"c8fbedcb-516b-4777-82c5-a50755673bff#7/2025-02-07/10-28-50.mp4",
"9826f220-2164-4434-aae0-b59e8de420b1#0/2025-02-06/10-00-47.mp4",
"1644dd53-210a-471f-b504-0cf4b14e5987#7/2025-02-06/09-55-27.mp4",
"e7489833-4f3c-460d-831d-6741932f219c#7/2025-01-03/14-15-05.mp4",
"448b900f-a3ed-48c5-b13a-6f889589e10e#0/2025-01-03/14-44-41.mp4",
"448b900f-a3ed-48c5-b13a-6f889589e10e#0/2025-01-03/15-44-42.mp4",
"448b900f-a3ed-48c5-b13a-6f889589e10e#0/2025-01-03/16-44-43.mp4",
"fe1b25e2-d187-45f8-89e7-2870880d07be#0/2025-01-02/10-47-53.mp4",
"264f26e3-3e1a-4b72-8401-7d586984faf3#7/2025-01-02/10-39-29.mp4",
"d8952213-ec4d-461f-9ec4-844004e3df3f#7/2024-12-31/14-06-13.mp4",
"1bb84174-b0d8-470e-be1c-3d90df1f7554#0/2024-12-31/14-18-35.mp4",
"d4e2fb4d-4d01-4a6c-8e1e-e4d961ffe9e0#1/2024-12-24/10-47-10.mp4",
"44a98a88-ed5b-4550-aa5f-1311d3aa5a56#1/2024-12-24/10-03-09.mp4"]
    files_list_size = len(files_list)
    print("files_list size: ", files_list_size)
    print("files_list [0:3]: ", files_list[0:3])

    count = 0
    for video_path in files_list:
        count += 1

        video_path = os.path.join(videos_source_dir, video_path)
        if not os.path.exists(video_path):
            print(f"input file: {video_path} didn't find")
            continue

        file_size = os.path.getsize(video_path)
        file_size_MB = file_size/1024/1024
        print(f"文件 {video_path} 的大小是 {file_size_MB} MB")
        

        out_file_name = get_out_file_name(video_path)
        if out_file_name == "":
            continue

        if os.path.exists(out_file_name):
            print(f"outfile {out_file_name} exists")
            continue
        
        print(f"begin one file:{out_file_name}")
        yolo_det_and_save_video(video_path, out_file_name)
        print(f"finish one file:{out_file_name}")
        
        print(f"进度: {count}/{files_list_size}")


        # exit()

        
        
    