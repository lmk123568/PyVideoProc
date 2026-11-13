import os

import torch

from .backend import TRTBackend


class YOLO26PoseTRT:
    def __init__(
        self,
        engine_path: str,
        conf_thres: float = 0.25,
        device: torch.device = torch.device("cuda:0"),
    ):
        self.conf_thres = conf_thres
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

        # postprocess(free nms)
        pred = pred[pred[..., 4] > self.conf_thres]
        pred[..., [0, 2]] = torch.clip(pred[..., [0, 2]], 0, self.w)
        pred[..., [1, 3]] = torch.clip(pred[..., [1, 3]], 0, self.h)
        pred[:, 6::3] = torch.clip(pred[:, 6::3], 0, self.w)
        pred[:, 7::3] = torch.clip(pred[:, 7::3], 0, self.h)

        return pred


def check():
    """
    精度对齐验证
    测试图片 assets/test_pose.jpg
    可视化结果 assets/test_pose_vis.jpg
    """
    import argparse

    import cv2
    import numpy as np

    from osd import draw_bbox, draw_pose
    from ultralytics import YOLO

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description="YOLO26 姿态精度对齐验证")
    parser.add_argument(
        "--engine", default="yolo26n-pose.engine", help="TensorRT engine 路径"
    )
    parser.add_argument(
        "--image",
        default=os.path.join(ROOT, "assets", "zidane.jpg"),
        help="测试图片路径",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(ROOT, "assets", "zidane_pose.jpg"),
        help="可视化结果保存路径",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    args = parser.parse_args()

    engine_path = os.path.abspath(args.engine)
    device = torch.device("cuda:0")

    # 1.加载模型
    ultra_model = YOLO(engine_path)
    our_model = YOLO26PoseTRT(engine_path, conf_thres=args.conf, device=device)

    # 2.测试图片
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"无法读取测试图片: {args.image}")
    img = cv2.resize(img, (our_model.w, our_model.h))

    img_bgr = torch.from_numpy(img).to(device)

    ultra_pred = ultra_model(img, conf=args.conf, verbose=False)[0]
    our_pred = our_model(img_bgr)

    ultra_boxes = ultra_pred.boxes
    our_boxes = our_pred.cpu()

    ultra_kpts = ultra_pred.keypoints.data.cpu()
    our_kpts = our_pred[:, 6:].reshape(our_pred.shape[0], -1, 3).cpu()

    print(f"==> Ultralytics \n{ultra_boxes.xyxy.cpu()}")
    print(f"==> PyVideoProc \n{our_boxes[:, :4]}")
    print(f"==> 检测框数量: Ultralytics={len(ultra_boxes)}, PyVideoProc={len(our_boxes)}")
    print(f"==> 关键点 shape: Ultralytics={tuple(ultra_kpts.shape)}, PyVideoProc={tuple(our_kpts.shape)}")

    if ultra_kpts.shape == our_kpts.shape:
        diff = (ultra_kpts[..., :2] - our_kpts[..., :2]).abs()
        print(f"==> 关键点 diff max_abs={float(diff.max()):.3e} mean_abs={float(diff.mean()):.3e}")

    # 3.可视化（上: Ultralytics 绿框, 下: PyVideoProc 红框）
    vis_ultra = img.copy()
    for box, conf, cls_id, kpts in zip(
        ultra_boxes.xyxy.cpu(),
        ultra_boxes.conf.cpu(),
        ultra_boxes.cls.cpu(),
        ultra_kpts,
    ):
        draw_bbox(
            vis_ultra,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color=(0, 255, 0),
            label=f"{float(conf):.2f} {int(cls_id)}",
        )
        draw_pose(vis_ultra, kpts.numpy(), color=(0, 255, 0))

    vis_our = img.copy()
    for box in our_boxes:
        draw_bbox(
            vis_our,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color=(64, 64, 255),
            label=f"{float(box[4]):.2f} {int(box[5])}",
        )
        draw_pose(vis_our, box[6:].reshape(-1, 3).numpy(), color=(64, 64, 255))

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
