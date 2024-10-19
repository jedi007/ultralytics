import cv2
from PIL import Image
from ultralytics import YOLO

if __name__ == '__main__': 
    # model = YOLO("hat_best.pt")
    # accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
    # results = model.predict(source="0")
    # results = model.predict(source="../datasets/test_image", show=True)  # Display preds. Accepts all YOLO predict arguments

    print("="*20)

    model = YOLO("helmet_241009.pt")
    im1 = cv2.imread("car.jpg")

    my_results = model(source=im1)
    # print("my_results: ", my_results)
    print("my_results.boxes: ", my_results[0].boxes)



    #获取视频设备/从视频文件中读取视频帧
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('/home/hyzh/lijie/data/test_video/1726132428485.mp4')

    #检测视频
    while cap.isOpened():
        #从摄像头读视频帧
        ret, frame = cap.read()

        frame = cv2.resize(frame, (720, 1280))

        if ret == True:
            results = model(source=frame, save=False, show=False)

            for r in results:
                # r.names = {0: 'personup', 1: 'persondown', 2: 'helmet', 3: 'nohelmet', 4: 'lanyard', 5: 'nolanyard'}
                # r.names = {0: '站立', 1: '跌倒', 2: '安全帽', 3: '未带安全帽', 4: '安全绳', 5: '未戴安全绳'}
                im_array = r.plot()  # plot a BGR numpy array of predictions
                # im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
                # im.show()  # show image
                # im.save('results.jpg')  # save image

                opencv_image = cv2.UMat(im_array)

                cv2.imwrite("test.jpg", opencv_image)


            # cv2.imshow("VideoCapture", opencv_image)
            
        #等待键盘事件，如果为q，退出
        key = cv2.waitKey(1)
        if(key & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            break
    
    cap.release()

    # results = model.predict(source=im1, save=True)  # save plotted images
    # print("results: ", results)

    exit(0)