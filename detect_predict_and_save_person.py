import cv2
from PIL import Image
from ultralytics import YOLO
import time


# prefix = int(time.time()/3600)
filename = "1726132428485.mp4"
prefix = filename
save_path = "/home/hyzh/lijie/data/test_out"
# save_path = "/home/hyzh/lijie/GitHub/V8/ultralytics/test_out"

if __name__ == '__main__': 
    model = YOLO("helmet_241009.pt")  # {0: 'personup', 1: 'persondown', 2: 'helmet', 3: 'nohelmet', 4: 'lanyard', 5: 'nolanyard'}
    im1 = cv2.imread("car.jpg")

    my_results = model(source=im1)
    # print("my_results: ", my_results)
    print("my_results.boxes: ", my_results[0].boxes)


    #获取视频设备/从视频文件中读取视频帧
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(f'/home/hyzh/lijie/data/test_video/{filename}')

    frame_index = -1
    #检测视频
    while cap.isOpened():
        #从摄像头读视频帧
        ret, frame = cap.read()

        if ret == False:  # 读取到最后一帧了
            break

        frame_index += 1
        if frame_index % 10 != 0:  # 每10帧检测一次
            continue

        if frame_index % 500 == 0:
            print(f"进度：frame_inde - {frame_index}")

        # frame = cv2.resize(frame, (720, 1280))

        if ret == True:
            results = model(source=frame, save=False, show=False)

            boxes = results[0].boxes

            for index in range(len(boxes.cls)):
                cls_id = int(boxes.cls[index])
                if cls_id != 0 and cls_id != 1: # 过滤掉不是人类的检测结果
                    continue
                
                box = boxes.xyxy[index]
                croped_img = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                
                save_name = f"{prefix}-{frame_index}-{index}.jpg"

                cv2.imwrite(f"{save_path}/{save_name}", croped_img)


            # im_array = results[0].plot()  # plot a BGR numpy array of predictions
            # opencv_image = cv2.UMat(im_array)
            # cv2.imwrite(f"{save_path}/1.jpg",opencv_image)

            # exit(0)

    
    cap.release()

    print("视频读取完毕，解析结束")

    exit(0)