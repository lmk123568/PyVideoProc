"""ByteTrack 的 numpy 批量加速版。

与 ``byte_tracker.py`` 接口完全一致（``ByteTracker(...).update(...)`` 返回
Nx7 数组），算法语义也完全一致，只把三处逐条运算换成批量运算：

  1. tlbr 批量转换  → 一帧里不再取 300 次属性，只算 3~4 次
  2. 卡尔曼 update 批量求逆 → N 次 4x4 Cholesky 变 1 次 batched solve
  3. multi_predict 向量化 → 去掉 Python 循环里的 N 次 np.diag

一致性验证见 ``bench_compare.py``。
"""

import numpy as np

from .basetrack import BaseTrack, TrackState
from .fast_ops import (
    KalmanFilterFast,
    fuse_score,
    iou_matrix,
    linear_assignment,
    tlbrs_of_tracks,
    xyahs_of,
)


class FastSTrack(BaseTrack):
    shared_kalman = KalmanFilterFast()

    def __init__(self, tlwh, score, cls_id=0):
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.score = score
        self.cls_id = cls_id
        self.tracklet_len = 0

    # -- 状态推进 ----------------------------------------------------------
    def activate(self, kalman_filter, frame_id):
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        # 复活是低频事件，保持单条路径，省得为了批量多写一套分支
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track._tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.cls_id = new_track.cls_id

    def apply_batch_update(self, mean, covariance, det, frame_id):
        """配合 ``KalmanFilterFast.batch_update`` 使用：线性代数已经算完，
        这里只负责把结果写回并推进状态字段。"""
        self.mean = mean
        self.covariance = covariance
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = det.score
        self.cls_id = det.cls_id

    def apply_batch_reactivate(self, mean, covariance, det, frame_id):
        """与 ``re_activate(new_id=False)`` 等价，但线性代数由批处理完成。"""
        self.mean = mean
        self.covariance = covariance
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.score = det.score
        self.cls_id = det.cls_id

    # -- 预测 --------------------------------------------------------------
    @staticmethod
    def multi_predict(stracks):
        if len(stracks) == 0:
            return
        means = np.stack([st.mean for st in stracks])          # (N,8)，stack 已经是新数组
        covs = np.stack([st.covariance for st in stracks])     # (N,8,8)
        # 非 Tracked 状态不外推高度变化（与原版一致，只清零 vh）
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                means[i, 7] = 0
        means, covs = FastSTrack.shared_kalman.batch_predict(means, covs)
        for i, st in enumerate(stracks):
            st.mean = means[i]
            st.covariance = covs[i]

    # -- 坐标转换 ----------------------------------------------------------
    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    def __repr__(self):
        return "OT_{}_({}-{})".format(self.track_id, self.start_frame, self.end_frame)


class FastByteTracker(object):
    def __init__(
        self,
        frame_rate=30,
        track_thresh=0.5,
        track_buffer=30,
        match_thresh=0.8,
        mot20=False,
    ):
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []

        self.frame_id = 0
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.mot20 = mot20
        self.det_thresh = track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilterFast()
        self.removed_buffer = 1000

    def reset(self):
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self.kalman_filter = KalmanFilterFast()
        self.reset_id()

    # ------------------------------------------------------------------
    def update(self, output_results, img_info=None, img_size=None):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # ---- 输入预处理 --------------------------------------------------
        if hasattr(output_results, "cpu"):
            output_results = output_results.cpu().numpy()
        output_results = np.ascontiguousarray(output_results, dtype=np.float32)
        if output_results.ndim == 1 or output_results.shape[0] == 0:
            output_results = np.empty((0, 6), dtype=np.float32)

        scores = output_results[:, 4]
        bboxes = output_results[:, :4].copy()
        cls_ids = (
            output_results[:, 5]
            if output_results.shape[1] >= 6
            else np.zeros(len(scores), dtype=np.float32)
        )

        if img_info is not None and img_size is not None:
            img_h, img_w = img_info[0], img_info[1]
            scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
            bboxes /= scale

        # ---- 按分数分两堆 ------------------------------------------------
        high = scores > self.track_thresh
        second = (scores > 0.1) & (scores < self.track_thresh)

        # 检测框的 tlbr 直接复用输入切片，省掉 tlbr→tlwh→tlbr 的往返
        dets_high_tlbr = bboxes[high]
        scores_high = scores[high]
        # 一次性把整批 tlbr 转成 tlwh（原版是每个框各做一次 numpy 运算）
        tlwh_high = dets_high_tlbr.copy()
        tlwh_high[:, 2:] -= tlwh_high[:, :2]
        det_high = [
            FastSTrack(tlwh_high[i], s, c)
            for i, (s, c) in enumerate(zip(scores_high, cls_ids[high]))
        ]

        dets_low_tlbr = bboxes[second]
        scores_low = scores[second]
        tlwh_low = dets_low_tlbr.copy()
        tlwh_low[:, 2:] -= tlwh_low[:, :2]
        det_low = [
            FastSTrack(tlwh_low[i], s, c)
            for i, (s, c) in enumerate(zip(scores_low, cls_ids[second]))
        ]

        # ---- 现有轨迹分家 ------------------------------------------------
        unconfirmed = []
        tracked_now = []
        for track in self.tracked_stracks:
            (tracked_now if track.is_activated else unconfirmed).append(track)

        # ---- Step 2：高分框 × (tracked + lost) ---------------------------
        pool = joint_stracks(tracked_now, self.lost_stracks)
        if pool:
            FastSTrack.multi_predict(pool)
        # 一次算好 tlbr，下面三步共用
        pool_tlbr = tlbrs_of_tracks(pool)

        cost = 1.0 - iou_matrix(pool_tlbr, dets_high_tlbr)
        if not self.mot20:
            cost = fuse_score(cost, scores_high)
        matches, u_track, u_det = linear_assignment(cost, thresh=self.match_thresh)

        to_update = []      # (pool_index, det_index) —— 需要卡尔曼修正
        to_refind = []      # 从 lost 里捞回来的
        for i, j in matches:
            if pool[i].state == TrackState.Tracked:
                to_update.append((i, j))
            else:
                to_refind.append((i, j))

        # 已经在跑的：批量修正
        if to_update:
            idx_t = np.fromiter((i for i, _ in to_update), dtype=np.intp, count=len(to_update))
            means = np.stack([pool[i].mean for i in idx_t])
            covs = np.stack([pool[i].covariance for i in idx_t])
            zs = xyahs_of([det_high[j] for _, j in to_update])
            means, covs = self.kalman_filter.batch_update(means, covs, zs)
            for k, (i, j) in enumerate(to_update):
                pool[i].apply_batch_update(means[k], covs[k], det_high[j], self.frame_id)
                activated_starcks.append(pool[i])

        # 从 lost 里捞回来的：同样批量修正（这条路径原版是逐条走 Cholesky）
        if to_refind:
            idx_t = np.fromiter((i for i, _ in to_refind), dtype=np.intp, count=len(to_refind))
            means = np.stack([pool[i].mean for i in idx_t])
            covs = np.stack([pool[i].covariance for i in idx_t])
            zs = xyahs_of([det_high[j] for _, j in to_refind])
            means, covs = self.kalman_filter.batch_update(means, covs, zs)
            for k, (i, j) in enumerate(to_refind):
                pool[i].apply_batch_reactivate(means[k], covs[k], det_high[j], self.frame_id)
                refind_stracks.append(pool[i])

        # ---- Step 3：低分框 × 上一步没配上的 tracked ----------------------
        # 这些轨迹本帧没被匹配，mean 仍是预测值，所以可以直接复用 pool_tlbr 的行
        r_idx = np.asarray(
            [i for i in to_int_array(u_track) if pool[i].state == TrackState.Tracked],
            dtype=np.intp,
        )
        r_tracked = [pool[i] for i in r_idx]
        if len(r_idx) and len(det_low) > 0:
            cost2 = 1.0 - iou_matrix(pool_tlbr[r_idx], dets_low_tlbr)
            m2, u2, _ = linear_assignment(cost2, thresh=0.5)
            pairs = [(i, j) for i, j in m2]
            if pairs:
                idx_t = np.fromiter((i for i, _ in pairs), dtype=np.intp, count=len(pairs))
                means = np.stack([r_tracked[i].mean for i in idx_t])
                covs = np.stack([r_tracked[i].covariance for i in idx_t])
                zs = xyahs_of([det_low[j] for _, j in pairs])
                means, covs = self.kalman_filter.batch_update(means, covs, zs)
                for k, (i, j) in enumerate(pairs):
                    r_tracked[i].apply_batch_update(means[k], covs[k], det_low[j], self.frame_id)
                    activated_starcks.append(r_tracked[i])
            for i in to_int_array(u2):
                r_tracked[i].mark_lost()
                lost_stracks.append(r_tracked[i])
        else:
            for track in r_tracked:
                track.mark_lost()
                lost_stracks.append(track)

        # ---- 未确认轨迹 × 剩下的高分框 -----------------------------------
        remain = to_int_array(u_det)
        det_remain = [det_high[i] for i in remain]
        det_remain_tlbr = dets_high_tlbr[remain] if len(remain) else np.empty((0, 4))
        scores_remain = scores_high[remain] if len(remain) else np.empty(0)

        if unconfirmed and det_remain:
            cost3 = 1.0 - iou_matrix(tlbrs_of_tracks(unconfirmed), det_remain_tlbr)
            if not self.mot20:
                cost3 = fuse_score(cost3, scores_remain)
            m3, u_unconf, u_det2 = linear_assignment(cost3, thresh=0.7)

            if len(m3):
                idx_t = np.fromiter((i for i, _ in m3), dtype=np.intp, count=len(m3))
                means = np.stack([unconfirmed[i].mean for i in idx_t])
                covs = np.stack([unconfirmed[i].covariance for i in idx_t])
                zs = xyahs_of([det_remain[j] for _, j in m3])
                means, covs = self.kalman_filter.batch_update(means, covs, zs)
                for k, (i, j) in enumerate(m3):
                    unconfirmed[i].apply_batch_update(means[k], covs[k], det_remain[j], self.frame_id)
                    activated_starcks.append(unconfirmed[i])

            for i in to_int_array(u_unconf):
                unconfirmed[i].mark_removed()
                removed_stracks.append(unconfirmed[i])
        else:
            for track in unconfirmed:
                track.mark_removed()
                removed_stracks.append(track)
            u_det2 = np.arange(len(det_remain), dtype=np.intp)

        # ---- Step 4：新建轨迹 --------------------------------------------
        for i in to_int_array(u_det2):
            track = det_remain[i]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        # ---- Step 5：清理超时的 lost -------------------------------------
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # ---- 合并状态 ----------------------------------------------------
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        if len(self.removed_stracks) > self.removed_buffer:
            self.removed_stracks = self.removed_stracks[-self.removed_buffer:]
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )

        output = [t for t in self.tracked_stracks if t.is_activated]
        if not output:
            return np.empty((0, 7))
        # 输出列：x1, y1, x2, y2, conf, cls, track_id
        out = np.empty((len(output), 7))
        out[:, :4] = tlbrs_of_tracks(output)
        out[:, 4] = [t.score for t in output]
        out[:, 5] = [t.cls_id for t in output]
        out[:, 6] = [t.track_id for t in output]
        return out


# ---------------------------------------------------------------------------
def to_int_array(idx):
    """linear_assignment 的未匹配索引在空矩阵时是 tuple，统一成 ndarray。"""
    return np.asarray(idx, dtype=np.intp).reshape(-1)


def joint_stracks(tlista, tlistb):
    a_ids = {t.track_id for t in tlista}
    return tlista + [t for t in tlistb if t.track_id not in a_ids]


def sub_stracks(tlista, tlistb):
    b_ids = {t.track_id for t in tlistb}
    return [t for t in tlista if t.track_id not in b_ids]


def remove_duplicate_stracks(stracksa, stracksb):
    if len(stracksa) == 0 or len(stracksb) == 0:
        return stracksa, stracksb
    pdist = 1.0 - iou_matrix(tlbrs_of_tracks(stracksa), tlbrs_of_tracks(stracksb))
    pairs = np.where(pdist < 0.15)
    dupa, dupb = set(), set()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.add(q)
        else:
            dupa.add(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb
