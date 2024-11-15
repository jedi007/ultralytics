import cv2
from PIL import Image
from ultralytics import YOLO
import time
import os
import random

# 设置随机数种子
random.seed(666)

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


def crop_obj(img, box, padding_scale = 0.1): 
    img_h, img_w, _ = img.shape
    
    x1 = box[0].item()
    x2 = box[2].item()
    y1 = box[1].item()
    y2 = box[3].item()

    box_w = x2 - x1
    box_h = y2 - y1

    w_padding = int(box_w * padding_scale)
    h_padding = int(box_h * padding_scale) * 2
    
    crop_x1 = max(x1 - w_padding, 0)
    crop_y1 = y1 # max(self.y1 - h_padding, 0)
    crop_x2 = min(x2 + w_padding, img_w)
    crop_y2 = min(y2 + h_padding, img_h)

    croped_img = img[int(crop_y1):int(crop_y2), int(crop_x1):int(crop_x2)]

    return croped_img


def crop_obj_add_random(img, box, padding_scale = 0.1):  
    img_h, img_w, _ = img.shape
    
    x1 = box[0].item()
    x2 = box[2].item()
    y1 = box[1].item()
    y2 = box[3].item()

    box_w = x2 - x1
    box_h = y2 - y1

    x1_padding = int(box_w * (padding_scale + random.uniform(0.01, 0.1)))
    y1_padding = int(box_h * random.uniform(0.01, 0.1))
    x2_padding = int(box_w * (padding_scale + random.uniform(0.01, 0.1)))
    y2_padding = int(box_h * (padding_scale + random.uniform(0.01, 0.1)) *2 )


    crop_x1 = max(x1 - x1_padding, 0)
    crop_y1 = max(y1 - y1_padding, 0)
    crop_x2 = min(x2 + x2_padding, img_w)
    crop_y2 = min(y2 + y2_padding, img_h)

    croped_img = img[int(crop_y1):int(crop_y2), int(crop_x1):int(crop_x2)]

    return croped_img


def one_img_crop(img_path, prefix, time_prefix):
    img = cv2.imread(img_path)

    results = model(source=img, save=False, show=False)

    boxes = results[0].boxes

    for obj_index in range(len(boxes.cls)):
        cls_id = int(boxes.cls[obj_index])
        if cls_id != 2 and cls_id != 3: # 2,3: helmet, nohelmet
            continue
        
        box = boxes.xyxy[obj_index]

        croped_img = crop_obj(img, box, 0.15)
        croped_img_random = crop_obj_add_random(img, box, 0.15)
                
        save_name = f"{prefix}-{obj_index}.jpg"
        save_name_random = f"{prefix}-{obj_index}-random.jpg"

        cv2.imwrite(f"{save_path}/{save_name}", croped_img)
        cv2.imwrite(f"{save_path}/{save_name_random}", croped_img_random)


time_prefix = int(time.time()/3600)
imgs_dir = "/home/hyzh/下载/抽烟打电话数据集/抽烟打电话-华录杯/train/train/normal"
save_path = "/home/hyzh/lijie/data/crop_out"
# save_path = "/home/hyzh/lijie/GitHub/V8/ultralytics/test_out"

if __name__ == '__main__': 
    model = YOLO("helmet_241009.pt")  # {0: 'personup', 1: 'persondown', 2: 'helmet', 3: 'nohelmet', 4: 'lanyard', 5: 'nolanyard'}
    
    im1 = cv2.imread("car.jpg")
    my_results = model(source=im1)
    # print("my_results: ", my_results)
    # print("my_results.boxes: ", my_results[0].boxes)

    file_path_list = traverse_folder(imgs_dir)
    file_name_list = traverse_folder_filename(imgs_dir)

    # print("file_path_list: ", file_path_list)

    count = 0
    for filename in file_name_list:
        file_path = f"{imgs_dir}/{filename}"
        
        one_img_crop(file_path, filename, time_prefix)

        count += 1

        if count % 100 == 0:
            print(f"进度： {count}")
        
        # if count == 10:
        #     break
