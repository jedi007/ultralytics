import cv2
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def yolo_pre():
    yolo = YOLO('det_personup_helmet_250107.pt')
    video_path = '/home/hyzh/lijie/cache/2024-03-14/10-30-42.mp4'
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    print("fps: ", fps)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('/home/hyzh/lijie/cache/2024-03-14/10-30-42_AI.mp4', fourcc, fps, (frame_width, frame_height))

    # 加载支持中文的字体，需要根据实际情况修改字体文件路径
    font_path = 'font.ttf'  # 例如使用黑体字体
    font = ImageFont.truetype(font_path, 20)

    # 定义类别名称和颜色映射
    class_names = {0: '工作服', 1: '跌倒', 2: '安全帽', 3: '未戴安全帽', 4: '安全绳', 5: '未戴安全绳'}
    color_map = {
        0: (255, 0, 0),  # 红色
        1: (0, 255, 0),  # 绿色
        2: (0, 0, 255),  # 蓝色
        3: (255, 255, 0),  # 黄色
        4: (255, 0, 255),  # 紫色
        5: (0, 255, 255)  # 青色
    }

    while cap.isOpened():
        status, frame = cap.read()
        if not status:
            break
        result = yolo.predict(source=frame, save=False, verbose=False)
        result = result[0]

        # 将 OpenCV 的 BGR 图像转换为 PIL 的 RGB 图像
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # 遍历检测结果，绘制中文标签和目标框
        for box in result.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            label = f"{class_names[class_id]} {conf:.2f}"
            color = color_map[class_id]

            # 绘制中文标签
            draw.text((x1, y1 - 20), label, font=font, fill=color)

            # 绘制目标框
            draw.rectangle((x1, y1, x2, y2), outline=color, width=2)

        # 将 PIL 图像转换回 OpenCV 的 BGR 图像
        anno_frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        out.write(anno_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print('保存完成')

yolo_pre()