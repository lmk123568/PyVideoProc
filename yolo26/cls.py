import os

import torch

from .backend import TRTBackend


class YOLO26ClsTRT:
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
        pred = pred.squeeze()

        return pred


def check():
    """
    精度对齐验证
    测试图片 assets/test_classify.jpg
    可视化结果 assets/test_classify_vis.jpg
    """
    import argparse

    import cv2

    from ultralytics import YOLO

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description="YOLO26 分类精度对齐验证")
    parser.add_argument(
        "--engine", default="yolo26n-cls.engine", help="TensorRT engine 路径"
    )
    parser.add_argument(
        "--image",
        default=os.path.join(ROOT, "assets", "zidane.jpg"),
        help="测试图片路径",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(ROOT, "assets", "zidane_cls.jpg"),
        help="可视化结果保存路径",
    )
    parser.add_argument("--topk", type=int, default=5, help="显示前 K 个类别")
    args = parser.parse_args()

    engine_path = os.path.abspath(args.engine)
    device = torch.device("cuda:0")

    # 1.加载模型
    ultra_model = YOLO(engine_path)
    our_model = YOLO26ClsTRT(engine_path, device=device)

    # 2.测试图片
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"无法读取测试图片: {args.image}")
    img_resized = cv2.resize(img, (our_model.w, our_model.h))

    img_bgr = torch.from_numpy(img_resized).to(device)

    ultra_probs = ultra_model(img_resized, verbose=False)[0].probs.data.cpu()
    our_probs = our_model(img_bgr).cpu()

    ultra_top5 = ultra_probs.topk(args.topk)
    our_top5 = our_probs.topk(args.topk)

    print(f"==> Ultralytics \n{ultra_top5}")
    print(f"==> PyVideoProc \n{our_top5}")

    if ultra_probs.shape == our_probs.shape:
        diff = (ultra_probs - our_probs).abs()
        print(f"==> 概率 diff max_abs={float(diff.max()):.3e} mean_abs={float(diff.mean()):.3e}")

    # 3.可视化（直接在图上写 top5）
    vis = img.copy()
    for i, (conf, cls_id) in enumerate(zip(our_top5.values, our_top5.indices)):
        cv2.putText(
            vis,
            f"Top{i + 1} {ultra_model.names[int(cls_id)]} {float(conf):.2%}",
            (20, 40 + i * 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(args.output, vis)
    print(f"==> 可视化结果已保存: {args.output}")


if __name__ == "__main__":
    check()
