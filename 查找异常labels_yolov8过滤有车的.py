import cv2
from PIL import Image
from ultralytics import YOLO
import os
import shutil

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


def file_path_exists(file_path):
    if os.path.exists(file_path):
        print(f"{file_path} 存在！")
    else:
        print(f"{file_path} 不存在！")
        b = os.mkdir(file_path)
        print(f"创建文件夹{file_path} b:{b}")

def load_label_boxes(label_path: str, width: int, height: int): # label file format: x,y,w,h 
    label_boxes = []
    for i in range(0,10):
        label_boxes.append([])
    

    if not os.path.exists(label_path):
        print(f"label文件不存在: {label_path}")
        return label_boxes

    # print("label_path: ", label_path)

    with open(label_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            args = line.strip().split()
            if len(args) == 5:
                cls_id = int(args[0])
                cx = float(args[1]) * width
                cy = float(args[2]) * height
                w = float(args[3]) * width
                h = float(args[4]) * height

                x1 = int(cx - w/2)
                x2 = int(cx + w/2)
                y1 = int(cy - h/2)
                y2 = int(cy + h/2)

                label_boxes[cls_id].append([x1, y1, x2, y2])
    
        file.close()
    
    return label_boxes

model = YOLO("yolov8s.pt")
def get_pred_boxes(img_path: str):
    im1 = Image.open(img_path)

    results = model.predict(source=im1, conf = 0.3)  # save plotted images

    boxes = results[0].boxes

    height, width = boxes.orig_shape

    pred_boxes = []
    for i in range(0,10):
        pred_boxes.append([])

    xyxys = boxes.xyxy.cpu().numpy()
    for obj_index in range(len(boxes.cls)):
        cls_id = int(boxes.cls[obj_index].item())

        if cls_id == 2 or cls_id == 5 or cls_id == 7: # 2,5,7 in coco dataset is car, bus,truck
            cls_id = 0
            pred_boxes[cls_id].append(xyxys[obj_index])
    
    return pred_boxes, width, height


names = {0:"car_plate", 1: "dangerous_plate", 2: "calling", 3: "other_clothes", 4: "no_reflective_jacket"}
colors = [(255,20,147), (106,90,205), (135,206,250), (60,179,113), (240,230,140), (255,99,71)]
colors = [(B,G,R) for (R,G,B) in colors]
show_size = (1600, 900)


import numpy as np
def show_boxes(img_path:str, boxes, title):
    # 确保读取中文路径
    file_bytes = np.fromfile(img_path, dtype=np.uint8)
    img = cv2.imdecode(file_bytes, -1) 
    # img = cv2.imread(image)

    for cls_id, boxes in enumerate(boxes):
        if len(boxes) == 0:
            continue

        name = names[cls_id]
        color=colors[cls_id]

        for box in boxes:
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.putText(img = img, text = name, org = (int(box[0]) + 5, int(box[1]) + 25), 
                fontFace = cv2.FONT_HERSHEY_PLAIN, fontScale = 2, color=color, thickness=2)
    
    img_s = cv2.resize(img, show_size)
    cv2.imshow(title, img_s)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

b_need_show_error_img = False

if __name__ == '__main__': 
    print("===========start")
    work_dir = R'''/home/hyzh/DATA/car_plate/source/2-org/2-org'''

    output_dir = f'''{work_dir}/out'''
    file_path_exists(output_dir)
    out_imgs_dir = os.path.join(output_dir, "images")
    out_labels_dir = os.path.join(output_dir, "labels")
    file_path_exists(out_imgs_dir)
    file_path_exists(out_labels_dir)

    imgs_dir = f"{work_dir}/images"
    labels_dir = f"{work_dir}/labels"
    file_path_list = traverse_folder_filename(imgs_dir)

    total_size = len(file_path_list)

    count = 0
    for img_name in file_path_list:
        img_path = os.path.join(imgs_dir, img_name)
        label_path = os.path.join(labels_dir, f"{img_name[0:-4]}.txt")

        pred_boxes, width, height = get_pred_boxes(img_path)
        # print("pred_boxes: ", pred_boxes)

        if len(pred_boxes[0]) == 0:
            continue
        
        shutil.move(img_path, out_imgs_dir)
        # shutil.move(label_path, out_labels_dir)

        count += 1
        if count % 100 == 2:
            print(f"进度: {count}/{total_size}")

        if count % 100 == 0:
            ss = "="*22
            print(f"{ss}over count: {count}")
