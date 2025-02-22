import cv2
from ultralytics import YOLO

def yolo_pre():
    yolo = YOLO('det_personup_helmet_250107.pt')
    video_path = '/home/hyzh/lijie/cache/2024-03-14/10-30-42.mp4'  # 检测视频的地址
    cap = cv2.VideoCapture(video_path)  # 创建一个VideoCapture对象，用于从视频文件中读取帧

    # 获取视频的帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("fps: ", fps)

    # 获取视频帧的维度
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # 创建VideoWriter对象，指定输出路径、编解码器、帧率和分辨率
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('/home/hyzh/lijie/cache/2024-03-14/10-30-42_AI.mp4', fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        status, frame = cap.read()  # 使用cap.read()从视频中读取每一帧
        if not status:
            break
        result = yolo.predict(source=frame, save=False, verbose = False)
        result = result[0]
        anno_frame = result.plot()
        out.write(anno_frame)  # 写入保存

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print('保存完成')

yolo_pre()