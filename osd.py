import cv2
import numpy as np


COCO_SKELETON = (
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 11), (6, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
)

def draw_bbox(
    image: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int] = (64, 64, 255),
    label: str = None,
) -> np.ndarray:
    x1, y1 = pt1
    x2, y2 = pt2

    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2, lineType=cv2.LINE_AA)

    if label is not None:
        w, h = cv2.getTextSize(
            label, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, thickness=2
        )[0]

        cv2.rectangle(
            image,
            (x1, y1 - h - 10),
            (x1 + w, y1),
            color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

        cv2.putText(
            image,
            text=label,
            org=(x1 + 1, y1 - 5),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.6,
            color=(0, 0, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    return image


def draw_pose(
    image: np.ndarray,
    kpts: np.ndarray,
    color: tuple[int, int, int] = (64, 64, 255),
    conf_thres: float = 0.0,
) -> np.ndarray:
    kpts = np.asarray(kpts, dtype=np.float32)

    for i, j in COCO_SKELETON:
        if kpts[i, 2] > conf_thres and kpts[j, 2] > conf_thres:
            cv2.line(
                image,
                (int(kpts[i, 0]), int(kpts[i, 1])),
                (int(kpts[j, 0]), int(kpts[j, 1])),
                color,
                thickness=2,
                lineType=cv2.LINE_AA,
            )

    for x, y, conf in kpts:
        if conf > conf_thres:
            cv2.circle(
                image,
                (int(x), int(y)),
                radius=4,
                color=color,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

    return image


def draw_mask(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    mask = np.asarray(mask).astype(np.int32)

    palette = cv2.applyColorMap(
        (np.arange(256) * 98 % 256).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_HSV
    ).reshape(256, 3)
    color = palette[mask]

    cv2.addWeighted(image, 1 - alpha, color, alpha, 0, dst=image)
    return image

