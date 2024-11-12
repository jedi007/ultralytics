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


def crop_obj(img, box, org = True):  # org 同时截取一张不随机扩展的原图
    img_h, img_w, _ = img.shape

    def random_change(v, hw, img_hw, model):
        # random_number = random.randint(1, 100)
        # if random_number < 60:
        change_v = hw / 10 # max(random.random() * hw / 10, 1)
        if model == "add":
            v += change_v
            v = min(v, img_hw)
        elif model == "sub":
            v -= change_v
            v = max(v, 0)
        
        return v
    
    x1 = box[0].item()
    x2 = box[2].item()
    y1 = box[1].item()
    y2 = box[3].item()

    if org == True:
        org_croped_img = img[int(y1):int(y2), int(x1):int(x2)]

    w = x2 - x1
    h = y2 - y1

    x1 = random_change(x1, w, img_w, "sub")
    y1 = random_change(y1, h, img_h, "sub")
    x2 = random_change(x2, w, img_w, "add")
    y2 = random_change(y2, h, img_h, "add")


    croped_img = img[int(y1):int(y2), int(x1):int(x2)]

    return croped_img, org_croped_img


def one_img_crop(img_path, prefix, time_prefix):
    img = cv2.imread(img_path)

    results = model(source=img, save=False, show=False)

    boxes = results[0].boxes

    for obj_index in range(len(boxes.cls)):
        cls_id = int(boxes.cls[obj_index])
        if cls_id != 2 and cls_id != 3: # 2,3: helmet, nohelmet
            continue
        
        box = boxes.xyxy[obj_index]

        croped_img, org_croped_img = crop_obj(img, box)
                
        save_name = f"{prefix}-{obj_index}.jpg"
        org_save_name = f"{prefix}-{obj_index}-org.jpg"

        cv2.imwrite(f"{save_path}/{save_name}", croped_img)
        cv2.imwrite(f"{save_path}/{org_save_name}", org_croped_img)


time_prefix = int(time.time()/3600)
imgs_dir = "/home/hyzh/下载/抽烟打电话数据集/抽烟打电话-华录杯/train/train/smoke"
save_path = "/home/hyzh/lijie/data/video_data/crop_out_smoke"
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
