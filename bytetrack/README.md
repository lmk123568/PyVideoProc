**ByteTrack 的纯 Numpy 批量加速优化实现**

匹配顺序、状态机和阈值不变，只把【逐条目标的 Python 循环 + 小数组运算】换成【整批 numpy 运算】，输出与原版逐帧一致（误差差 2e-3），测试加速 1.5 ~ 2.1 倍

原版：[Fundation/ByteTrack](https://github.com/FoundationVision/ByteTrack)

**快速开始**

```bash
pip install numpy lap
```

```python
from bytetrack.fast_byte_tracker import FastByteTracker

tracker = ByteTracker(
    frame_rate=30,         # 推理帧率
    track_thresh=0.5,      # 高低分分界
    track_buffer=30,       # 丢失后保留帧数
    match_thresh=0.8,      # 第一次匹配的代价阈值
)

for frame in video:
    dets = detector(frame)          # (N, 6) = x1, y1, x2, y2, score, cls
    online = tracker.update(dets)   # (M, 7) = x1, y1, x2, y2, score, cls, track_id
```

**优化改动**

|  | 改动 | 原版写法 | 加速版写法 | 效果 |
|---|---|---|---|---|
| 1 | tlbr 批量转换 | `[t.tlbr for t in tracks]`，每条建两个数组，一帧约 300 次 | `tlbrs_of_tracks()` 一次 `np.stack` 出 `(N,4)`，一帧 5 次 | 5.9x |
| 2 | 卡尔曼 update 批量化 | 每条轨迹单独走 4x4 Cholesky（`cho_factor` + `cho_solve`） | 一次 batched `np.linalg.solve` 处理 `(N,4,4)`；协方差修正为 `(N,8,8)` | 17.9x |
| 3 | `re_activate` 也批量化 | 从 lost 找回的轨迹逐条走 Cholesky | 与 #2 共用同一条批量路径 | 省 ~50 µs/帧 |
| 4 | `multi_predict` 向量化 | Python 循环 `np.diag` 造 N 个 8x8 阵；`dot + transpose` 两次 | `sqr[:,:,None] * eye8` 广播；`F @ P @ F.T` | 4.1x |
| 5 | `fuse_score` 去 repeat | `np.expand_dims(...).repeat(N, 0)` 实打实分配 N×M 内存 | 广播 `scores[None, :]` | 1.3x |
| 6 | 检测框坐标复用 | 每个框单独做 `tlbr → tlwh`，共 2 次 numpy 调用 | 整批一次 `tlwh = tlbr.copy(); tlwh[:,2:] -= tlwh[:,:2]` | 省 ~15% |
| 7 | 去重加空判 + 切片复用 | `lost` 为空时也算整张 IoU 矩阵 | 空了直接返回；Step 3 复用 `pool_tlbr[r_idx]` 切片 | 省 ~30 µs/帧 |

**Benchmark**

MOT17 train 标注 + 官方公开检测，7 个序列（02/04/05/09/10/11/13），每个变体 5316 帧

| 检测源 | 跟踪器 | MOTA   | MOTP   | IDF1   | FP   | FN    | IDSW | 速度                    |
| ------ | ------ | ------ | ------ | ------ | ---- | ----- | ---- | ----------------------- |
| FRCNN  | 原版   | 0.4934 | 0.8721 | 0.5484 | 1627 | 54591 | 672  | 1.61 ms/帧              |
| FRCNN  | 改进版 | 0.4934 | 0.8721 | 0.5484 | 1627 | 54591 | 672  | 0.82 ms/帧（**1.96x**） |
| SDP    | 原版   | 0.6387 | 0.8505 | 0.6305 | 1927 | 37643 | 1002 | 1.95 ms/帧              |
| SDP    | 改进版 | 0.6387 | 0.8505 | 0.6305 | 1927 | 37643 | 1002 | 0.93 ms/帧（**2.10x**） |
| DPM    | 原版   | 0.2739 | 0.7910 | 0.3506 | 1087 | 80099 | 358  | 1.08 ms/帧              |
| DPM    | 改进版 | 0.2739 | 0.7910 | 0.3506 | 1087 | 80099 | 358  | 0.73 ms/帧（**1.47x**） |
