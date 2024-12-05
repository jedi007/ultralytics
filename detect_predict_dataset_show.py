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
    model = YOLO("det_dangerous_plate_241205.pt")
    print("="*20)

    imgs_dir = "/home/hyzh/DATA/car_plate/source/检测到危险牌out-筛选/dangerousplate/images"
    file_path_list = traverse_folder_filename(imgs_dir)

    count = 0
    for img_name in file_path_list:
        count += 1
        if count % 20 != 0:
            continue

        img_path = os.path.join(imgs_dir, img_name)

        im1 = Image.open(img_path)

        results = model.predict(source=im1, conf = 0.3)  # save plotted images

        # print("result[0].boxes: ", results[0].boxes)
        
        # results = model(source=frame, save=False, show=False)

        for r in results:
            # r.names = {0: 'personup', 1: 'persondown', 2: 'helmet', 3: 'nohelmet', 4: 'lanyard', 5: 'nolanyard'}
            # r.names = {0: '站立', 1: '跌倒', 2: '安全帽', 3: '未带安全帽', 4: '安全绳', 5: '未戴安全绳'}
            im_array = r.plot()  # plot a BGR numpy array of predictions
            # im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            # im.show()  # show image
            # im.save('results.jpg')  # save image

            opencv_image = cv2.UMat(im_array)

            # cv2.imwrite("test.jpg", opencv_image)

            cv2.imshow("show_img_results", opencv_image)
            
            #等待键盘事件，如果为q，退出
            key = cv2.waitKey(0)
            if(key & 0xFF == ord('q')):
                cv2.destroyAllWindows()
                exit()

        if count % 100 == 0:
            ss = "="*22
            print(f"{ss}over count: {count}")
