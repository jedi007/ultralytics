from pathlib import Path

from ultralytics.utils.torch_utils import strip_optimizer

# ===================== 可配置参数（统一管理，方便修改）=====================
SOURCE_WEIGHTS = "runs/pose/train19/weights/epoch162.pt"
OUTPUT_WEIGHTS = "runs/pose/train19/weights/epoch162_stripped.pt"
# =========================================================================

if __name__ == "__main__":
    src = Path(SOURCE_WEIGHTS)
    dst = Path(OUTPUT_WEIGHTS)

    if not src.exists():
        raise FileNotFoundError(f"找不到输入权重: {src}")
    if dst.resolve() == src.resolve():
        raise ValueError("输出路径不能与输入文件相同")

    strip_optimizer(src, dst)
    print(f"输入: {src} ({src.stat().st_size / 1e6:.1f} MB)")
    print(f"输出: {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
