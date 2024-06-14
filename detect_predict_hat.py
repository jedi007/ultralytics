import cv2
from PIL import Image
from ultralytics import YOLO

if __name__ == '__main__': 
    model = YOLO("hat_best.pt")
    # accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
    # results = model.predict(source="0")
    # results = model.predict(source="../datasets/test_image", show=True)  # Display preds. Accepts all YOLO predict arguments

    print("="*20)

    im1 = cv2.imread("E:/TestData/hat_1.png")

    my_results = model(source=im1)
    print("my_results: ", my_results)
    print("my_results.boxes: ", my_results[0].boxes)




    # results = model.predict(source=im1, save=True)  # save plotted images
    # print("results: ", results)

    exit(0)