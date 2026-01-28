import torch
import torch.nn.functional as F

from .backend import TRTBackend


class YOLO26DetTRT:
    def __init__(
        self,
        weights: str,
        conf_thres: float,
        device: torch.device = torch.device("cuda:0"),
    ):
        self.conf_thres = conf_thres
        self.backend = TRTBackend(engine_path=weights, device=device)

    def __call__(self, im: torch.Tensor) -> torch.Tensor:
        assert im.shape == (
            576,
            1024,
            3,
        ), "[YOLO26] image shape must be (576, 1024, 3)"

        # preprocess
        im = im.permute(2, 0, 1)
        im = im[None]
        im = im.contiguous()
        im = im.float()
        im /= 255

        # inference
        pred = self.backend(im)

        # postprocess(no nms)
        pred = pred[pred[..., 4] > self.conf_thres]

        # clip bboxes
        pred[..., [0, 2]] = torch.clip(pred[..., [0, 2]], 0, 1024)
        pred[..., [1, 3]] = torch.clip(pred[..., [1, 3]], 0, 576)

        return pred


class YOLO26SegTRT:
    pass


class YOLO26ClsTRT:
    pass
