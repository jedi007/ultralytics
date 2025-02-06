import cv2
from PIL import Image
from ultralytics import YOLO

if __name__ == '__main__': 
    model = YOLO("det_dangerousplate_250122_2.pt")
    # accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
    # results = model.predict(source="0")
    # results = model.predict(source="../datasets/test_image", show=True)  # Display preds. Accepts all YOLO predict arguments

    print("="*20)

    im1 = Image.open("test2.jpg")

    my_results = model(source=im1)
    print("my_results: ", my_results)

    results = model.predict(source=im1, save=True)  # save plotted images
    print("results: ", results)


    # from PIL
    im1 = Image.open("../datasets/test_image/bus.jpg")
    results = model.predict(source=im1, save=True)  # save plotted images

    # from ndarray
    im2 = cv2.imread("../datasets/test_image/zidane.jpg")
    results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels

    # from list of PIL/ndarray
    # results = model.predict(source=[im1, im2])