from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from ultralytics.utils import TQDM

# =========================
# 宏定义配置区（直接修改）
# =========================
VIDEO_PATH = "/home/robot/github/test_code/JKGN/大华/save_videos/rtsp_record_2026-08-05-16-10-55.mp4"
OUTPUT_ROOT_DIR = "/data/清洗cache/caiji/video_pred2"
OUTPUT_FRAME_DIR = f"{OUTPUT_ROOT_DIR}/frames"

# 截取控制参数
start_index = 100    # 从第几帧开始处理
frames_count = 100   # 需要保存的图片总数量

# 画面相似度过滤配置
FRAME_SIMILARITY_THRESHOLD = 0.95
HIST_COMPARE_METHOD = cv2.HISTCMP_CORREL
RESIZE_SIMILARITY_W = 640
RESIZE_SIMILARITY_H = 384

# =========================
# 工具函数
# =========================
def calc_gray_hist(frame: np.ndarray, w: int, h: int) -> np.ndarray:
    """计算灰度直方图用于相似度对比"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (w, h))
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist

def main():
    # 提取视频文件名（不带后缀）
    video_file = Path(VIDEO_PATH)
    video_stem = video_file.stem
    # 创建输出文件夹
    save_dir = Path(OUTPUT_FRAME_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件：{VIDEO_PATH}")

    ref_hist = None
    current_frame_idx = 0
    saved_num = 0
    skip_similar = 0
    total_scan = 0

    pbar = TQDM(desc="视频帧截取", unit="帧")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            current_frame_idx += 1
            total_scan += 1
            pbar.update(1)

            # 1. 跳过起始帧之前的所有画面
            if current_frame_idx < start_index:
                continue

            # 2. 达到目标保存数量，直接退出
            if saved_num >= frames_count:
                break

            # 3. 相似度判断
            curr_hist = calc_gray_hist(frame, RESIZE_SIMILARITY_W, RESIZE_SIMILARITY_H)
            if ref_hist is not None:
                sim_score = cv2.compareHist(ref_hist, curr_hist, HIST_COMPARE_METHOD)
                if sim_score >= FRAME_SIMILARITY_THRESHOLD:
                    skip_similar += 1
                    continue

            # 画面发生变化，更新参考直方图 + 保存图片
            ref_hist = curr_hist
            # 文件名格式：视频文件名_帧序号.jpg
            save_name = f"{video_stem}_{current_frame_idx:06d}.jpg"
            save_path = save_dir / save_name
            cv2.imwrite(str(save_path), frame)
            saved_num += 1

    finally:
        pbar.close()
        cap.release()

    # 打印统计信息
    print("=" * 55)
    print(f"视频路径: {VIDEO_PATH}")
    print(f"起始截取帧: {start_index} | 目标保存数: {frames_count}")
    print(f"相似度阈值: {FRAME_SIMILARITY_THRESHOLD}")
    print(f"总共扫描帧数: {total_scan}")
    print(f"相似重复帧跳过: {skip_similar}")
    print(f"实际保存图片数量: {saved_num}")
    print(f"图片保存目录: {save_dir.resolve()}")
    print("=" * 55)

if __name__ == "__main__":
    main()