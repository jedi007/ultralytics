import sys
import os
import shutil

# 获取上一级目录的绝对路径
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 将上一级目录添加到 sys.path 中
sys.path.append(parent_dir)


from ultralytics import YOLO
import cv2


def traverse_folder(folder_path):
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            abs_file_path = os.path.join(root, file_name)
            # print(abs_file_path)
            file_list.append(abs_file_path)
    
    return file_list

def file_path_exists(file_path):
    if os.path.exists(file_path):
        print(f"{file_path} 存在！")
    else:
        print(f"{file_path} 不存在！")
        b = os.mkdir(file_path)
        print(f"创建文件夹{file_path} b:{b}")

class_names_array = ['other_clothes', 'security', 'unknow', 'work_clothes']
dataset_dir = "/data/train_data/cls_working_clothes/val"
out_dir = "/data/train_data/cls_working_clothes/val_error_out"

file_path_exists(out_dir)

if __name__ == '__main__':  
    # 加载模型
    model = YOLO("./weights/test/cls_work_clothes_e15_250225.pt")  # {0: 'other_clothes', 1: 'security', 2: 'unknow', 3: 'work_clothes'}

    # im1 = cv2.imread("car.jpg")

    # cls_result = model(source=im1)  
    # print("cls_result: ", cls_result)
    # print("cls_result probs: ", cls_result[0].probs)
    # print("cls_result[0].probs.top1: ", cls_result[0].probs.top1)

    for id, class_name in enumerate(class_names_array):
        print(f"id: {id}, type: {type(id)}")
        print("class_name: ", class_name)

        class_dir = os.path.join(dataset_dir, class_name)
        print(f"class_dir: {class_dir}")

        image_path_list = traverse_folder(class_dir)
        image_path_list_size = len(image_path_list)
        print(f"image_path_list size: {image_path_list_size}")

        class_out_dir = os.path.join(out_dir, class_name)
        file_path_exists(class_out_dir)

        count = 0
        move_count = 0
        for image_path in image_path_list:
            count += 1
            if count % 500 == 0:
                print(f"{class_name} 进度: {count}/{image_path_list_size}")

            img = cv2.imread(image_path)
            cls_result = model.predict(source=img, save=False, verbose=False)  
            pred_id = cls_result[0].probs.top1

            if pred_id != id:
                shutil.move(image_path, class_out_dir)
                move_count += 1
        
        print(f"{class_name} move out: {move_count}")

    