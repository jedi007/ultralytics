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

def load_label_boxes(label_path: str, width: int, height: int):
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

model = YOLO("det_dangerous_plate_241205.pt")
def get_pred_boxes(img_path: str):
    im1 = Image.open(img_path)

    results = model.predict(source=im1, conf = 0.3)  # save plotted images

    boxes = results[0].boxes
    # print("boxes: ", boxes)

    height, width = boxes.orig_shape

    pred_boxes = []
    for i in range(0,10):
        pred_boxes.append([])

    xyxys = boxes.xyxy.cpu().numpy()
    for obj_index in range(len(boxes.cls)):
        cls_id = int(boxes.cls[obj_index].item())

        pred_boxes[cls_id].append(xyxys[obj_index])
    
    return pred_boxes, width, height

def checkout_boxes_number(pred_boxes, label_boxes):
    for i in range(0, len(pred_boxes)):
        if len(pred_boxes[i]) != len(label_boxes[i]):
            return False
    
    return True

# 将预测的box和label的box进行一一匹配，一一都匹配上了则返回True.  当出现有大部分面积重叠的box时，可能会匹配错误。暂时忽略了这种情况。  计算iou匹配表可解决该问题，但计算量偏大
def check_boxes_iou(pred_boxes, label_boxes, iou_threshold):  # 低于阈值的都算作预测错误  
    def calculate_box_iou(box1, box2): # box: x1, y1, x2, y2
        # Intersection area
        inter_w = max((min(box1[2], box2[2]) - max(box1[0], box2[0])), 0)
        inter_h = max((min(box1[3], box2[3]) - max(box1[1], box2[1])), 0)
        inter = inter_w * inter_h

        def calculate_box_area(box):
            return (box[2] - box[0]) * (box[3] - box[1])

        eps = 1e-7
        union = calculate_box_area(box1) + calculate_box_area(box2) - inter + eps

        iou = inter / union

        return iou
    
    for cls_id in range(0, len(pred_boxes)):
        if len(pred_boxes[cls_id]) == 0:
            continue
        
        for box1 in pred_boxes[cls_id]:
            b_matched = False
            for box2 in label_boxes[cls_id]:
                iou = calculate_box_iou(box1, box2)

                if iou > iou_threshold:
                    label_boxes[cls_id].remove(box2)  # 从label中删除已被匹配上的Box
                    b_matched = True
                    break
            
            if b_matched == False:
                return False
    
    return True


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
    work_dir = R'''/home/hyzh/DATA/car_plate/source/筛选-危险牌/dangerousplate'''

    output_dir = f'''{work_dir}/out'''
    file_path_exists(output_dir)
    out_imgs_dir = os.path.join(output_dir, "images")
    out_labels_dir = os.path.join(output_dir, "labels")
    file_path_exists(out_imgs_dir)
    file_path_exists(out_labels_dir)

    imgs_dir = f"{work_dir}/images"
    labels_dir = f"{work_dir}/labels"
    file_path_list = traverse_folder_filename(imgs_dir)


    count = 0
    for img_name in file_path_list:
        img_path = os.path.join(imgs_dir, img_name)
        label_path = os.path.join(labels_dir, f"{img_name[0:-4]}.txt")

        pred_boxes, width, height = get_pred_boxes(img_path)
        # print("pred_boxes: ", pred_boxes)

        label_boxes = load_label_boxes(label_path, width, height)
        # print("label_boxes: ", label_boxes)
        
        b_error_label = False
        if not checkout_boxes_number(pred_boxes, label_boxes):
            print(f"pred_boxes, label_boxes shape 不相等 : {img_name}")
            b_error_label = True

        if not b_error_label and not check_boxes_iou(pred_boxes, label_boxes, 0.7):
            print(f"label iou error : {img_name}")
            b_error_label = True
        else:
            print(f"pred and label matched : {img_name}")
        
        if b_error_label:
            if b_need_show_error_img:
                show_boxes(img_path, pred_boxes, "pred_boxes")
                show_boxes(img_path, label_boxes, "label_boxes")
                key = cv2.waitKey(0)
                if(key & 0xFF == ord('q')):
                    exit()
                # cv2.destroyAllWindows()

            shutil.move(img_path, out_imgs_dir)
            shutil.move(label_path, out_labels_dir)

        if count % 100 == 0:
            ss = "="*22
            print(f"{ss}over count: {count}")
