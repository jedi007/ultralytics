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


def one_img_crop(img_path, prefix, time_prefix):
    img = cv2.imread(img_path)

    results = model(source=img, save=False, show=False)

    boxes = results[0].boxes

    for obj_index in range(len(boxes.cls)):
        cls_id = int(boxes.cls[obj_index])
        if cls_id != 0 and cls_id != 1: # 过滤掉不是人类的检测结果
            continue
        
        box = boxes.xyxy[obj_index]
        croped_img = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        
        save_name = f"{prefix}-{obj_index}.jpg"

        cv2.imwrite(f"{save_path}/{save_name}", croped_img)


time_prefix = int(time.time()/3600)
imgs_dir = "/home/hyzh/lijie/data/images_data/images"
save_path = "/home/hyzh/lijie/data/images_data/crop_out"
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

    for filename in file_name_list:
        file_path = f"{imgs_dir}/{filename}"
        
        one_img_crop(file_path, filename, time_prefix)
