# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
from __future__ import annotations

import argparse
import random

import cv2
import numpy as np
import onnxruntime as ort
import torch

from ultralytics.utils import ASSETS, YAML
from ultralytics.utils.checks import check_requirements, check_yaml


class YOLO26:
    """YOLO26 object detection model class for handling ONNX inference (带NMS的ONNX) and visualization."""

    def __init__(self, onnx_model: str, input_image: str, confidence_thres: float, iou_thres: float):
        """Initialize YOLO26, 修复：固定随机种子保证颜色一致"""
        self.onnx_model = onnx_model
        self.input_image = input_image
        self.confidence_thres = confidence_thres  # 带NMS的ONNX可二次过滤置信度
        self.iou_thres = iou_thres

        # Load the class names from the COCO dataset
        self.classes = YAML.load(check_yaml("coco8.yaml"))["names"]
        # 修复1：固定随机种子，避免每次运行类别颜色不同
        random.seed(42)
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

        # 新增：初始化输入尺寸（由ONNX模型自动获取，无需提前定义）
        self.input_width = None
        self.input_height = None
        self.img = None
        self.img_height = None
        self.img_width = None

    def letterbox(self, img: np.ndarray, new_shape: tuple[int, int] = (640, 640)) -> tuple[np.ndarray, tuple[int, int]]:
        """Resize and pad image, 原代码无问题，直接复用"""
        shape = img.shape[:2]  # current shape [height, width]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = round(shape[1] * r), round(shape[0] * r)
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img, (top, left)

    def draw_detections(self, img: np.ndarray, box: list[float], score: float, class_id: int) -> None:
        """绘制检测框，修复2：适配x1y1x2y2格式（原代码是x1y1wh格式）"""
        # 修复：输入框为x1,y1,x2,y2，直接解包
        x1, y1, x2, y2 = box
        color = self.color_palette[class_id]
        # 绘制矩形框（x1y1到x2y2）
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        # 标签文字
        label = f"{self.classes[class_id]}: {score:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # 标签位置（避免超出图片顶部）
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_h else y1 + 10

        # 核心修复：裁剪标签背景的坐标，避免超出图片边界
        img_h, img_w = img.shape[:2]
        label_x = max(0, min(label_x, img_w - label_w))  # 标签左边界不越界
        label_y = max(label_h, min(label_y, img_h - label_h))  # 标签上下不越界

        # 绘制标签背景和文字
        cv2.rectangle(img, (label_x, label_y - label_h), (label_x + label_w, label_y + label_h), color, cv2.FILLED)
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def preprocess(self) -> tuple[np.ndarray, tuple[int, int]]:
        """预处理图片，原代码无问题，直接复用"""
        self.img = cv2.imread(self.input_image)
        assert self.img is not None, f"无法读取图片：{self.input_image}"  # 新增：图片读取校验
        self.img_height, self.img_width = self.img.shape[:2]
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        img, pad = self.letterbox(img, (self.input_height, self.input_width))
        image_data = np.array(img) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))  # CHW
        image_data = np.expand_dims(image_data, 0).astype(np.float32)  # (1,3,H,W)
        return image_data, pad

    # 新增辅助函数：坐标还原+边界裁剪（核心，解决框位置错误）
    def scale_coords(self, box: np.ndarray, pad: tuple[int, int]) -> np.ndarray:
        """
        将ONNX输出的坐标（640×640，含padding）还原为原图坐标，并裁剪到图片边界
        :param box: 单框坐标 [x1, y1, x2, y2]（np.ndarray）
        :param pad: letterbox的padding (top, left)
        :return: 原图坐标 [x1, y1, x2, y2]（np.ndarray）
        """
        # 步骤1：减去padding（消除letterbox的填充）
        box[0] -= pad[1]  # x1减left
        box[1] -= pad[0]  # y1减top
        box[2] -= pad[1]  # x2减left
        box[3] -= pad[0]  # y2减top
        # 步骤2：计算缩放比例（原图尺寸 / letterbox后的有效尺寸）
        gain = min(self.input_height / self.img_height, self.input_width / self.img_width)
        # 步骤3：还原为原图坐标
        box /= gain
        # 步骤4：裁剪坐标，避免超出图片边界（核心修复：解决框超出图片的问题）
        box[0] = np.clip(box[0], 0, self.img_width)   # x1 ≥0 且 ≤原图宽
        box[1] = np.clip(box[1], 0, self.img_height)  # y1 ≥0 且 ≤原图高
        box[2] = np.clip(box[2], 0, self.img_width)   # x2 ≥0 且 ≤原图宽
        box[3] = np.clip(box[3], 0, self.img_height)  # y2 ≥0 且 ≤原图高
        return box

    def postprocess(self, output: list[np.ndarray], pad: tuple[int, int]) -> np.ndarray:
        """
        后处理核心函数：适配带NMS的ONNX输出，修复所有解析BUG
        输入：ONNX输出、letterbox的padding
        输出：绘制好检测框的原图
        """
        # 步骤1：解析带NMS的输出 → shape=(300,6) [x1,y1,x2,y2,conf,cls_id]
        outputs = np.squeeze(output[0])  # 去掉批次维度，(max_det,6)
        print(f"带NMS的ONNX输出shape：{outputs.shape}")  # 应输出 (300,6)

        # 步骤2：过滤无效框（置信度≥设定阈值，带NMS的ONNX已过滤0置信度，这里做二次过滤）
        valid_mask = outputs[:, 4] >= self.confidence_thres
        valid_dets = outputs[valid_mask]  # 有效检测框，shape=(n,6)，n为实际检测数
        print(f"有效检测框数量：{len(valid_dets)}")

        # 步骤3：遍历有效框，还原坐标+绘制
        img_copy = self.img.copy()  # 避免修改原图
        for det in valid_dets:
            x1, y1, x2, y2, conf, cls_id = det
            # 修复3：正确提取类别ID（转为整数，原代码错误提取第4位为类别ID）
            cls_id = int(cls_id)
            # 坐标还原+边界裁剪
            box_ori = self.scale_coords(np.array([x1, y1, x2, y2]), pad)
            box_ori = box_ori.astype(np.int32)
            # 绘制检测框
            self.draw_detections(img_copy, box_ori, conf, cls_id)

        return img_copy

    def main(self) -> np.ndarray:
        """主推理函数，微小调整：提前获取ONNX输入尺寸"""
        # 加载ONNX模型，自动选择CUDA/CPU
        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        session = ort.InferenceSession(self.onnx_model, providers=providers or available)

        # 获取ONNX输入尺寸（自动适配，无需硬编码640）
        model_inputs = session.get_inputs()
        input_shape = model_inputs[0].shape  # (1,3,H,W)
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]
        print(f"ONNX模型输入尺寸：{self.input_width}×{self.input_height}")

        # 预处理+推理
        img_data, pad = self.preprocess()
        outputs = session.run(None, {model_inputs[0].name: img_data})

        # 后处理（修复：移除多余的input_image参数，原代码传参冗余）
        return self.postprocess(outputs, pad)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolo26n.onnx", help="带NMS的YOLO26 ONNX模型路径")
    parser.add_argument("--img", type=str, default=str(R"E:\code\github_code\testdata\1.jpg"), help="输入图片路径")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="置信度二次过滤阈值（建议0.25）")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="带NMS的ONNX无需此参数，保留兼容")
    args = parser.parse_args()

    # 检查依赖
    check_requirements("onnxruntime-gpu" if torch.cuda.is_available() else "onnxruntime")

    # 推理+可视化
    detection = YOLO26(args.model, args.img, args.conf_thres, args.iou_thres)
    output_image = detection.main()

    # 显示结果（自适应窗口大小）
    cv2.namedWindow("YOLO26 ONNX Detection (with NMS)", cv2.WINDOW_NORMAL)
    cv2.imshow("YOLO26 ONNX Detection (with NMS)", output_image)
    cv2.waitKey(0)
    # 新增：保存结果图片（可选）
    cv2.imwrite("detection_result.jpg", output_image)
    cv2.destroyAllWindows()