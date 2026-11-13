import os

import torch

from .backend import TRTBackend


class YOLO26SemTRT:
    def __init__(
        self,
        engine_path: str,
        device: torch.device = torch.device("cuda:0"),
    ):
        self.backend = TRTBackend(engine_path=engine_path, device=device)
        _, _, self.h, self.w = self.backend.bindings["images"].shape

    def __call__(self, im: torch.Tensor) -> torch.Tensor:

        if im.shape != (self.h, self.w, 3):
            raise RuntimeError(f"im.shape must be ({self.h}, {self.w}, 3)")

        # preprocess
        im = im.permute(2, 0, 1)
        im = im[[2, 1, 0]]
        im = im[None]
        im = im.contiguous()
        im = im.float()
        im /= 255

        # inference
        pred = self.backend(im)

        # postprocess
        pred = pred.squeeze(0)
        if pred.ndim == 3:
            pred = pred.argmax(0)

        return pred


def check():
    """
    精度对齐验证
    测试图片 assets/test_semantic.jpg
    可视化结果 assets/test_semantic_vis.jpg
    """
    import argparse

    import cv2
    import numpy as np

    from osd import draw_mask
    from ultralytics import YOLO

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description="YOLO26 语义分割精度对齐验证")
    parser.add_argument(
        "--engine", default="yolo26n-sem.engine", help="TensorRT engine 路径"
    )
    parser.add_argument(
        "--image",
        default=os.path.join(ROOT, "assets", "zidane.jpg"),
        help="测试图片路径",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(ROOT, "assets", "zidane_sem.jpg"),
        help="可视化结果保存路径",
    )
    args = parser.parse_args()

    engine_path = os.path.abspath(args.engine)
    device = torch.device("cuda:0")

    # 1.加载模型
    ultra_model = YOLO(engine_path)
    our_model = YOLO26SemTRT(engine_path, device=device)

    # 2.测试图片
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"无法读取测试图片: {args.image}")
    img = cv2.resize(img, (our_model.w, our_model.h))

    img_bgr = torch.from_numpy(img).to(device)

    ultra_pred = ultra_model(img, verbose=False)[0]
    our_pred = our_model(img_bgr)

    ultra_mask = ultra_pred.semantic_mask.data.cpu().numpy().astype(np.int64)
    our_mask = our_pred.cpu().numpy().astype(np.int64)

    print(f"==> mask shape: Ultralytics={ultra_mask.shape}, PyVideoProc={our_mask.shape}")
    print(f"==> 类别: Ultralytics={len(np.unique(ultra_mask))}, PyVideoProc={len(np.unique(our_mask))}")

    if ultra_mask.shape == our_mask.shape:
        diff = np.abs(ultra_mask - our_mask)
        acc = float((ultra_mask == our_mask).mean())
        print(f"==> mask diff max_abs={float(diff.max()):.3e} mean_abs={float(diff.mean()):.3e}")
        print(f"==> mask 一致率={acc:.4%}")

    # 3.可视化（上: Ultralytics, 下: PyVideoProc）
    vis_ultra = img.copy()
    draw_mask(vis_ultra, ultra_mask)

    vis_our = img.copy()
    draw_mask(vis_our, our_mask)

    for panel, title in ((vis_ultra, "Ultralytics"), (vis_our, "PyVideoProc")):
        cv2.putText(
            panel,
            title,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    vis = np.vstack([vis_ultra, vis_our])
    cv2.imwrite(args.output, vis)
    print(f"==> 可视化结果已保存: {args.output}")


if __name__ == "__main__":
    check()
