# ssh root@203.25.213.139    pw:8K9mRMERPpubwM8z   /data/zlm/data/www/record
# scp -r live root@203.25.213.139:/data/zlm/data/www/record

import json
from copy import deepcopy
import re
import os
 
current_path = os.path.dirname(os.path.realpath(__file__))

print("current_path: ", current_path)

videos_source_dir = "/data/cache/live0"

out_dir = "/data/cache/live_add_AI_from_list2"

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


    files_list = []
    with open('600G_filelist.txt', 'r', encoding='utf-8') as list_file:
        print("open successed")
        for line in list_file:
            # line = rtsp_file.readline().replace("\\n","")
            if len(line) > 1:
                files_list.append(line.replace("\n",""))

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

        # continue

        # if file_size_MB > 200:
        #     continue
        

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

        
        
    