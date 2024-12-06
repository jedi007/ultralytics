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

    print("label_path: ", label_path)

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
            for box2 in label_boxes[cls_id]:
                iou = calculate_box_iou(box1, box2)

                if iou > iou_threshold:
                    


if __name__ == '__main__': 
    print("===========start")

    imgs_dir = "/home/hyzh/DATA/car_plate/source/筛选-危险牌/dangerousplate/images"
    labels_dir = "/home/hyzh/DATA/car_plate/source/筛选-危险牌/dangerousplate/labels"
    file_path_list = traverse_folder_filename(imgs_dir)

    count = 0
    for img_name in file_path_list:
        img_path = os.path.join(imgs_dir, img_name)
        label_path = os.path.join(labels_dir, f"{img_name[0:-4]}.txt")

        pred_boxes, width, height = get_pred_boxes(img_path)
        print("pred_boxes: ", pred_boxes)

        label_boxes = load_label_boxes(label_path, width, height)
        print("label_boxes: ", label_boxes)
        
        if not checkout_boxes_number(pred_boxes, label_boxes):
            print("pred_boxes, label_boxes shape 不相等")

        check_boxes_iou(pred_boxes, label_boxes, 0.7)
        exit()

            



        if count % 100 == 0:
            ss = "="*22
            print(f"{ss}over count: {count}")
