import cv2
from PIL import Image
from ultralytics import YOLO
import os
import shutil
from copy import deepcopy

model = YOLO("det_personup_helmet_241119.pt")

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


def get_pred_boxes(img_path: str, filter_classes):
    im1 = Image.open(img_path)

    results = model.predict(source=im1, conf = 0.2)  # save plotted images

    boxes = results[0].boxes
    # print("boxes: ", boxes)

    height, width = boxes.orig_shape

    pred_boxes = []
    for i in range(0,10):
        pred_boxes.append([])

    xyxys = boxes.xyxy.cpu().numpy()
    for obj_index in range(len(boxes.cls)):
        cls_id = int(boxes.cls[obj_index].item())

        if cls_id in filter_classes:
            pred_boxes[cls_id].append(xyxys[obj_index])
    
    return pred_boxes, width, height

def checkout_boxes_number(pred_boxes, label_boxes, filter_classes):
    for i in range(0, len(pred_boxes)):
        if i in filter_classes:
            if len(pred_boxes[i]) != len(label_boxes[i]):
                return False
    
    return True

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

# 将预测的box和label的box进行一一匹配
def check_boxes_iou(pred_boxes, label_boxes, iou_threshold, filter_classes):  # 有一个匹配上就算通过 
    all_empty = True
    for cls_id in filter_classes:
        if len(pred_boxes[cls_id]) == 0 and len(label_boxes[cls_id]) == 0:
            continue

        all_empty = False

        for box1 in pred_boxes[cls_id]:
            for box2 in label_boxes[cls_id]:
                iou = calculate_box_iou(box1, box2)

                if iou > iou_threshold:
                    return True
    
    if all_empty == True:
        return True
    else:
        return False


names = {0:"car_plate", 1: "dangerous_plate", 2: "calling", 3: "other_clothes", 4: "no_reflective_jacket"}
colors = [(255,20,147), (106,90,205), (135,206,250), (60,179,113), (240,230,140), (255,99,71)]
colors = [(B,G,R) for (R,G,B) in colors]
show_size = (1600, 900)

def xyxy2xywhn(box, width, height):
    w = box[2] - box[0]
    h = box[3] - box[1]
    cx = box[0] + w/2
    cy = box[1] + h/2

    return cx/width, cy/height, w/width, h/height 

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


def merge_boxes(label_boxes, pred_boxes, iou_threshold = 0.5):  # 高于阈值的进行删除，不合并
    def box_is_in_boxes(box1, boxes, iou_threshold):
        for box2 in boxes:
            iou = calculate_box_iou(box1, box2)
            if iou > iou_threshold:
                return True
        
        return False

    merged_boxes = deepcopy(label_boxes)
    
    for cls_id, boxes in enumerate(pred_boxes):
        for box in boxes:
            if not box_is_in_boxes(box, label_boxes[cls_id], iou_threshold):
                merged_boxes[cls_id].append(box)
    
    return merged_boxes



b_need_show_error_img = False
merge_label_then_out = True  # 融合label Box 和 pred box 到新的label文件q
filter_classes = [0] # 仅对关注的类别进行处理


#用于查找标注和推理一个都匹配不上的labels
if __name__ == '__main__': 
    print("===========start")
    work_dir = R'''/home/hyzh/DATA/train_data/det_data_add_zsgl/train'''

    output_dir = f'''{work_dir}/error_label_out'''
    file_path_exists(output_dir)
    out_imgs_dir = os.path.join(output_dir, "images")
    out_labels_dir = os.path.join(output_dir, "labels")
    file_path_exists(out_imgs_dir)
    file_path_exists(out_labels_dir)

    imgs_dir = f"{work_dir}/images"
    labels_dir = f"{work_dir}/labels"
    file_path_list = traverse_folder_filename(imgs_dir)

    print("files: ", file_path_list[0:3])

    total_size = len(file_path_list)

    count = 0
    for img_name in file_path_list:
        img_path = os.path.join(imgs_dir, f"{img_name[0:-4]}.jpg")
        label_path = os.path.join(labels_dir, f"{img_name[0:-4]}.txt")

        if not os.path.exists(img_path):
            continue

        print("img_path: ", img_path)

        pred_boxes, width, height = get_pred_boxes(img_path, filter_classes)
        # print("pred_boxes: ", pred_boxes)

        label_boxes = load_label_boxes(label_path, width, height)
        # print("label_boxes: ", label_boxes)



        if not check_boxes_iou(pred_boxes, deepcopy(label_boxes), 0.5, filter_classes):
            print(f"label iou error : {img_name}")
        
            if b_need_show_error_img:
                show_boxes(img_path, pred_boxes, "pred_boxes")
                show_boxes(img_path, label_boxes, "label_boxes")
                key = cv2.waitKey(0)
                if(key & 0xFF == ord('q')):
                    exit()
                # cv2.destroyAllWindows()

            count += 1
            if count % 100 == 2:
                print(f"进度: {count}/{total_size}")
            
            # if count == 10:
            #     exit()

            if merge_label_then_out:
                out_label_path = os.path.join(out_labels_dir, f"{img_name[0:-4]}.txt")

                merged_boxes = merge_boxes(label_boxes, pred_boxes, 0.5)

                labels_str = ""
                for cls_id, boxes in enumerate(merged_boxes):
                    for box in boxes:
                        cx, cy, w, h = xyxy2xywhn(box, width, height)
                        if labels_str == "":
                            labels_str = f"{cls_id} {cx} {cy} {w} {h}"
                        else:
                            labels_str += f"\n{cls_id} {cx} {cy} {w} {h}"
                
                with open(out_label_path, 'w') as file:
                    file.write(labels_str)
                    file.close()
                    # print("write labels: ", labels_str)
                    # exit()

                shutil.move(img_path, out_imgs_dir)
            else:
                shutil.move(label_path, out_labels_dir)
                shutil.move(img_path, out_imgs_dir)
