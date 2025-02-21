import cv2
from PIL import Image
from ultralytics import YOLO
import time
import os
import random
import numpy as np

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

def save_img(save_path, image):
    # 编码图像
    _, img_encoded = cv2.imencode('.jpg', image)
    
    # 将编码后的图像数据写入文件
    with open(save_path, 'wb') as f:
        f.write(img_encoded.tobytes())

def one_img_crop(img_path, prefix, time_prefix):
    with open(img_path, 'rb') as f:
        # 读取文件内容
        img_bytes = f.read()
        # 将读取的字节数据转换为 numpy 数组
        img_array = np.frombuffer(img_bytes, np.uint8)
        # 使用 cv2.imdecode 解码图像数组
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # img = cv2.imread(img_path)
    height, width, channels = img.shape

    results = model(source=img, save=False, show=False, verbose = False)

    boxes = results[0].boxes

    for obj_index in range(len(boxes.cls)):
        cls_id = int(boxes.cls[obj_index])
        if cls_id != 0 and cls_id != 1: # 过滤掉不是人类的检测结果
            continue
        
        box = boxes.xyxy[obj_index]
        x1,y1,x2,y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        croped_img = img[y1:y2, x1:x2]

        # save other_clothes crop
        clothes_result = model_cls_clothes(source=croped_img, save=False, show=False, verbose = False)
        cls_id = clothes_result[0].probs.top1

        if cls_id == 0:
            save_name = f"{prefix}-{obj_index}.jpg"
            # cv2.imwrite(f"{save_path}/{save_name}", croped_img)
            save_img(f"{save_otherclothes_path}/{save_name}", croped_img)

            save_aug_name = f"{prefix}-{obj_index}_aug.jpg"
            save_path = os.path.join(save_otherclothes_aug_path, save_aug_name)
            save_aug_cropimg(save_path, img, width, height, x1, y1, x2, y2)
        
        
        
        #  save ref crop
        ref_result = model_cls_refjacket(source=croped_img, save=False, show=False, verbose = False)
        cls_id = ref_result[0].probs.top1
        
        if cls_id == 0:
            save_name = f"{prefix}-{obj_index}.jpg"
            # cv2.imwrite(f"{save_path}/{save_name}", croped_img)
            save_img(f"{save_no_ref_path}/{save_name}", croped_img)

            save_aug_name = f"{prefix}-{obj_index}_aug.jpg"
            save_path = os.path.join(save_no_ref_aug_path, save_aug_name)
            save_aug_cropimg(save_path, img, width, height, x1, y1, x2, y2)



def save_aug_cropimg(save_path, img, width, height, x1, y1, x2, y2):
        croped_w = x2 - x1
        croped_h = y2 - y1

        # 生成一个 2 到 5 之间的随机浮点数
        random_float = random.uniform(1, 5) / 100
        aug_x1 = max(int(x1 - croped_w*random_float), 0)
        random_float = random.uniform(1, 5) / 100
        aug_x2 = min(int(x2 + croped_w*random_float), width)
        
        random_float = random.uniform(1, 5) / 100
        aug_y1 = max(int(y1 - croped_h*random_float), 0)
        random_float = random.uniform(1, 5) / 100
        aug_y2 = min(int(y2 + croped_h*random_float), height)
        
        img_augmentation = img[aug_y1:aug_y2, aug_x1:aug_x2]
        
        # cv2.imwrite(f"{save_aug_path}/{save_aug_name}", img_augmentation)
        save_img(save_path, img_augmentation)



def file_path_exists(file_path):
    if os.path.exists(file_path):
        print(f"{file_path} 存在！")
    else:
        print(f"{file_path} 不存在！")
        b = os.mkdir(file_path)
        print(f"创建文件夹{file_path} b:{b}")

time_prefix = int(time.time()/3600)
imgs_dir = "/data/保安分类/images"
save_path = "/data/保安分类/crop_out"
save_aug_path = "/data/保安分类/crop_out_aug"
file_path_exists(save_path)
file_path_exists(save_aug_path)

save_otherclothes_path = os.path.join(save_path, 'otherclothes')
save_otherclothes_aug_path = os.path.join(save_aug_path, 'otherclothes')
file_path_exists(save_otherclothes_path)
file_path_exists(save_otherclothes_aug_path)

save_no_ref_path = os.path.join(save_path, 'no_ref')
save_no_ref_aug_path = os.path.join(save_aug_path, 'no_ref')
file_path_exists(save_no_ref_path)
file_path_exists(save_no_ref_aug_path)

if __name__ == '__main__': 
    model = YOLO("det_personup_helmet_241119.pt")  # {0: 'personup', 1: 'persondown', 2: 'helmet', 3: 'nohelmet', 4: 'lanyard', 5: 'nolanyard'}

    model_cls_clothes = YOLO("cls_workclothes_241123_s_e12.pt")  # {0: 'other_clothes', 1: 'security', 2: 'unknow', 3: 'work_clothes'}

    model_cls_refjacket = YOLO("cls_refjacket_241122_s_e14.pt") # {0: 'no_reflective_jacket', 1: 'reflective_jacket', 2: 'unknow'}
    
    im1 = cv2.imread("car.jpg")
    my_results = model(source=im1)
    # print("my_results: ", my_results)
    # print("my_results.boxes: ", my_results[0].boxes)

    cls_result = model_cls_clothes(source=im1)  
    # print("cls_result: ", cls_result)
    print("cls_result probs: ", cls_result[0].probs)
    print("cls_result[0].probs.top1: ", cls_result[0].probs.top1)

    cls_ref_result = model_cls_refjacket(source=im1)  
    # print("cls_ref_result: ", cls_ref_result)



    file_path_list = traverse_folder(imgs_dir)
    file_name_list = traverse_folder_filename(imgs_dir)

    # print("file_path_list: ", file_path_list)

    count = 0
    total_size = len(file_name_list)
    for filename in file_name_list:
        file_path = f"{imgs_dir}/{filename}"
        
        one_img_crop(file_path, filename, time_prefix)

        count += 1
        # if count == 10:
        #     break
        if count % 100 == 0:
            print(f"进度： {count}/{total_size}")
