"""ByteTrack 的 numpy 批量加速原语。

原实现的热点不在算法，而在「同一个量被反复逐条计算」：

  1. ``STrack.tlbr`` 是 property，每次访问都新建两个 numpy 数组。
     而 ``iou_distance`` 在 ``update()`` 里出现 5 处（Step 2 / Step 3 /
     unconfirmed / 去重 / 输出），一帧下来同一条轨迹的 tlbr 被算 3~5 遍。
  2. ``KalmanFilter.update`` 每条轨迹单独走一遍 4x4 Cholesky 求解。
  3. ``multi_predict`` 用 Python 循环构造 N 个 8x8 对角阵。

本模块把它们换成三件事：批量转换、批量线性代数、广播代替显式复制。
对外只提供函数和一个小类，不改变任何算法语义（逐位一致性见
``bench_compare.py`` 的对比测试）。
"""

import numpy as np

try:
    import lap
except ImportError as exc:  # pragma: no cover
    raise ImportError("需要 lap 包：pip install lap") from exc

from .kalman_filter import KalmanFilter

_EYE8 = np.eye(8)
_DIAG4 = np.arange(4)


# ---------------------------------------------------------------------------
# 边界框批量转换
# ---------------------------------------------------------------------------
def tlbrs_of_tracks(stracks):
    """list[STrack] -> (N,4) float64 的 tlbr。

    替代 ``[st.tlbr for st in stracks]``。要求每条轨迹都已经有 ``mean``
    （即已 activate 过），tracked / lost / unconfirmed 都满足。
    """
    n = len(stracks)
    if n == 0:
        return np.empty((0, 4), dtype=np.float64)
    m = np.stack([st.mean for st in stracks])          # (N,8)
    w = m[:, 2] * m[:, 3]                              # 宽 = 宽高比 * 高
    x1 = m[:, 0] - w * 0.5
    y1 = m[:, 1] - m[:, 3] * 0.5
    out = np.empty((n, 4), dtype=np.float64)
    out[:, 0] = x1
    out[:, 1] = y1
    out[:, 2] = x1 + w
    out[:, 3] = y1 + m[:, 3]
    return out


def tlbrs_of_dets(detections):
    """list[STrack] -> (N,4) float64 的 tlbr，用于还没进滤波器的检测框。

    走 ``_tlwh`` 而不是 ``mean``，替代 ``[d.tlbr for d in detections]``。
    """
    n = len(detections)
    if n == 0:
        return np.empty((0, 4), dtype=np.float64)
    out = np.stack([d._tlwh for d in detections]).astype(np.float64, copy=True)
    out[:, 2:] += out[:, :2]
    return out


def xyahs_of(detections):
    """list[STrack] -> (N,4) xyah，替代 ``[d.to_xyah() for d in detections]``。"""
    n = len(detections)
    if n == 0:
        return np.empty((0, 4), dtype=np.float64)
    out = np.stack([d._tlwh for d in detections]).astype(np.float64, copy=True)
    out[:, 0] += out[:, 2] * 0.5
    out[:, 1] += out[:, 3] * 0.5
    out[:, 2] /= out[:, 3]
    return out


# ---------------------------------------------------------------------------
# IoU 距离
# ---------------------------------------------------------------------------
def iou_matrix(a, b):
    """(N,4) x (M,4) 的 tlbr -> (N,M) 的 IoU 矩阵。"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[None, :, 0], b[None, :, 1], b[None, :, 2], b[None, :, 3]
    iw = np.minimum(ax2, bx2) - np.maximum(ax1, bx1)
    ih = np.minimum(ay2, by2) - np.maximum(ay1, by1)
    np.maximum(iw, 0.0, out=iw)
    np.maximum(ih, 0.0, out=ih)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / np.maximum(union, 1e-9)


def iou_distance(a_tlbr, b_tlbr):
    """等价于 ``matching.iou_distance``，但输入是预先算好的 (N,4) / (M,4)。"""
    return 1.0 - iou_matrix(a_tlbr, b_tlbr)


def fuse_score(cost_matrix, det_scores):
    """等价于 ``matching.fuse_score``，但分数用数组传入、用广播代替 repeat。

    原实现 ``np.expand_dims(...).repeat(N, 0)`` 会实打实分配一整块 N×M 内存，
    只是为了把它当常量广播用。
    """
    if cost_matrix.size == 0:
        return cost_matrix
    return 1.0 - (1.0 - cost_matrix) * det_scores[None, :]


# ---------------------------------------------------------------------------
# 匹配
# ---------------------------------------------------------------------------
def linear_assignment(cost_matrix, thresh):
    """与 ``matching.linear_assignment`` 语义完全一致。"""
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            tuple(range(cost_matrix.shape[0])),
            tuple(range(cost_matrix.shape[1])),
        )
    matches = []
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    return np.asarray(matches), np.where(x < 0)[0], np.where(y < 0)[0]


# ---------------------------------------------------------------------------
# 卡尔曼滤波批量版
# ---------------------------------------------------------------------------
class KalmanFilterFast(KalmanFilter):
    """在原有 KalmanFilter 上补充两个批量接口，单条接口保持不变。"""

    def batch_predict(self, mean, covariance):
        """mean (N,8)  covariance (N,8,8) -> (N,8), (N,8,8)。

        两处改动：
          - motion_cov 由 N 次 ``np.diag`` 改成一次广播乘法
          - ``dot(F, P).transpose + dot(left, F.T)`` 改成 ``F @ P @ F.T``
        """
        if len(mean) == 0:
            return mean, covariance
        h = mean[:, 3]                       # 用预测前的 h 构造过程噪声，与原版一致
        wp = self._std_weight_position
        wv = self._std_weight_velocity
        std_pos = np.stack([wp * h, wp * h, np.full_like(h, 1e-2), wp * h], axis=1)
        std_vel = np.stack([wv * h, wv * h, np.full_like(h, 1e-5), wv * h], axis=1)
        motion_cov = np.square(np.concatenate([std_pos, std_vel], axis=1))[:, :, None] * _EYE8

        f = self._motion_mat
        return mean @ f.T, f @ covariance @ f.T + motion_cov

    def batch_update(self, mean, covariance, measurements):
        """mean (N,8)  covariance (N,8,8)  measurements (N,4) -> (N,8), (N,8,8)。

        把 N 次 4x4 Cholesky 求解合并成一次 batched ``np.linalg.solve``。
        观测噪声 R 是逐条轨迹不同的（依赖各自的 h），所以直接把 R 的对角
        加到投影协方差上，不显式构造 N 个 4x4 矩阵。
        """
        if len(mean) == 0:
            return mean, covariance
        h = mean[:, 3:4]
        wh = self._std_weight_position * h
        std = np.concatenate([wh, wh, np.full_like(h, 1e-1), wh], axis=1)   # (N,4)

        h_mat = self._update_mat
        proj_mean = mean @ h_mat.T                       # (N,4)
        proj_cov = h_mat @ covariance @ h_mat.T          # (N,4,4)
        proj_cov[:, _DIAG4, _DIAG4] += std * std         # 等价于 + R

        cov_ht = covariance @ h_mat.T                    # (N,8,4)
        # K = cov @ H.T @ S^-1，用 batched solve 代替求逆
        k = np.linalg.solve(proj_cov, cov_ht.transpose(0, 2, 1)).transpose(0, 2, 1)

        innovation = (measurements - proj_mean)[:, :, None]          # (N,4,1)
        new_mean = mean + (k @ innovation)[:, :, 0]                  # (N,8)
        new_cov = covariance - k @ proj_cov @ k.transpose(0, 2, 1)   # (N,8,8)
        return new_mean, new_cov
