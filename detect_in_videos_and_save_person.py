import cv2
from PIL import Image
from ultralytics import YOLO
import time
import os

# 设置一个放vidoe 文件的文件夹路径作为输入
# 设置一个放裁剪好的图片的路径作为输出
# 运行后会把输入文件夹中的视频文件全部解析并把裁剪的全部图片放在输出文件夹中

def traverse_folder(folder_path):
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            abs_file_path = os.path.join(root, file_name)
            # print(abs_file_path)
            file_list.append(abs_file_path)
    
    return file_list

def traverse_folder_filename(folder_path):
    filename_list = []
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            filename_list.append(file_name)
    
    return filename_list


def cap_video_crop(video_path, prefix):
    #获取视频设备/从视频文件中读取视频帧
    # cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture(f'/home/hyzh/lijie/data/test_video/{filename}')
    cap = cv2.VideoCapture(video_path)

    frame_index = -1
    #检测视频
    while cap.isOpened():
        #从摄像头读视频帧
        ret, frame = cap.read()

        if ret == False:  # 读取到最后一帧了
            break

        frame_index += 1
        if frame_index % 8 != 0:  # 每10帧检测一次
            continue

        if frame_index % 500 == 0:
            print(f"{video_path}:进度：frame_inde - {frame_index}")

        # frame = cv2.resize(frame, (720, 1280))

        if ret == True:
            results = model(source=frame, save=False, show=False)

            boxes = results[0].boxes

            for obj_index in range(len(boxes.cls)):
                cls_id = int(boxes.cls[obj_index])
                if cls_id != 0 and cls_id != 1: # 过滤掉不是人类的检测结果
                    continue
                
                box = boxes.xyxy[obj_index]
                croped_img = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                
                save_name = f"{prefix}-{frame_index}-{obj_index}.jpg"

                cv2.imwrite(f"{save_path}/{save_name}", croped_img)

    
    cap.release()

    print(f"{video_path} 视频读取完毕，解析裁剪结束")


# prefix = int(time.time()/3600)
video_base_path = "/home/hyzh/lijie/data/video_data/bilibili_video"
save_path = "/home/hyzh/lijie/data/video_data/crop_out"
# save_path = "/home/hyzh/lijie/GitHub/V8/ultralytics/test_out"

if __name__ == '__main__': 
    model = YOLO("helmet_241009.pt")  # {0: 'personup', 1: 'persondown', 2: 'helmet', 3: 'nohelmet', 4: 'lanyard', 5: 'nolanyard'}
    
    im1 = cv2.imread("car.jpg")
    my_results = model(source=im1)
    # print("my_results: ", my_results)
    # print("my_results.boxes: ", my_results[0].boxes)

    video_file_path_list = traverse_folder(video_base_path)
    video_file_name_list = traverse_folder_filename(video_base_path)

    print("video_file_name_list: ", video_file_name_list)

    for filename in video_file_name_list:
        file_path = f"{video_base_path}/{filename}"
        prefix = filename

        cap_video_crop(file_path, prefix)

    exit(0)