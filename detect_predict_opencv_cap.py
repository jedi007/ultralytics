import cv2
from PIL import Image
from ultralytics import YOLO

if __name__ == '__main__': 
    # model = YOLO("hat_best.pt")
    # accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
    # results = model.predict(source="0")
    # results = model.predict(source="../datasets/test_image", show=True)  # Display preds. Accepts all YOLO predict arguments

    print("="*20)

    model = YOLO("yolov8n.pt")
    im1 = cv2.imread("E:/TestData/hat_1.png")

    my_results = model(source=im1)
    print("my_results: ", my_results)
    print("my_results.boxes: ", my_results[0].boxes)



    #获取视频设备/从视频文件中读取视频帧
    cap = cv2.VideoCapture(0)

    #检测视频
    while cap.isOpened():
        #从摄像头读视频帧
        ret, frame = cap.read()

        if ret == True:
            results = model(source=frame, save=False, show=False)

            for r in results:
                im_array = r.plot()  # plot a BGR numpy array of predictions
                # im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
                # im.show()  # show image
                # im.save('results.jpg')  # save image

                opencv_image = cv2.UMat(im_array)

            cv2.imshow("VideoCapture", opencv_image)
            
        #等待键盘事件，如果为q，退出
        key = cv2.waitKey(1)
        if(key & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            break
    
    cap.release()

    # results = model.predict(source=im1, save=True)  # save plotted images
    # print("results: ", results)

    exit(0)