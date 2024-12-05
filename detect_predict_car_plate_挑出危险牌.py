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

if __name__ == '__main__': 
    model = YOLO("det_dangerousplate_241204.pt")
    # accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
    # results = model.predict(source="0")
    # results = model.predict(source="/home/hyzh/DATA/car_plate/car_plate_det_data_241203/val/images", show=False, save=True)  # Display preds. Accepts all YOLO predict arguments

    # "/home/hyzh/DATA/car_plate/car_plate_det_data_241203/val/images/川GE3L05_821.274780&435.841370_1023.110840&441.414825_1024.132690&517.330261_821.906616&511.591919.jpg"

    print("="*20)

    imgs_dir = "/home/hyzh/DATA/car_plate/car_plate_det_data_241203/train/images"
    file_path_list = traverse_folder_filename(imgs_dir)

    out_dir = "/home/hyzh/DATA/car_plate/car_plate_det_data_241203/train/out"
    file_path_exists(out_dir)

    out_imgs = os.path.join(out_dir, "images")
    out_labels = os.path.join(out_dir, "labels")
    file_path_exists(out_imgs)
    file_path_exists(out_labels)

    count = 0
    for img_name in file_path_list:
        img_path = os.path.join(imgs_dir, img_name)

        im1 = Image.open(img_path)

        results = model.predict(source=im1, conf=0.01)  # save plotted images
        boxes = results[0].boxes

        b_dangerous = False
        for obj_index in range(len(boxes.cls)):
            cls_id = int(boxes.cls[obj_index])
            if cls_id != 1:
                continue
            
            b_dangerous = True

        if b_dangerous:
            label_txt = ""
            for obj_index in range(len(boxes.cls)):
                cls_id = int(boxes.cls[obj_index])
                if cls_id != 1 and cls_id != 0:
                    continue

                if label_txt == "":
                    label_txt += f"{int(boxes.cls[obj_index])} {str(boxes.xywhn[obj_index].cpu().numpy())[1:-1]}"
                else:
                     label_txt += f"\n{int(boxes.cls[obj_index])} {str(boxes.xywhn[obj_index].cpu().numpy())[1:-1]}"
            
            print(f"label_txt: \n{label_txt}")

            shutil.move(img_path, out_imgs)

            label_path = os.path.join(out_labels, f"{img_name[0:-4]}.txt")

            with open(label_path, 'w') as file:
                file.write(label_txt)
                file.close()

            #exit()

        count += 1
        if count % 100 == 0:
            ss = "="*22
            print(f"{ss}over count: {count}")
