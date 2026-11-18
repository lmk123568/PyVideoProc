import os
import time

# 多进程单线程
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp

import osd
from bytetrack.fast_byte_tracker import FastByteTracker
from cpp import Decoder, Encoder
from yolo26.det import YOLO26DetTRT


class Pipeline(mp.Process):
    def __init__(self, gpu, input_url, output_url, engine_path):
        super().__init__()
        self.gpu = gpu
        self.input_url = input_url
        self.output_url = output_url
        self.engine_path = engine_path

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    def run(self):
        # 模型初始化
        model_det = YOLO26DetTRT(
            engine_path=self.engine_path,
            conf_thres=0.25,
            device=torch.device("cuda:0"),
        )
        model_w, model_h = model_det.w, model_det.h

        # 解码初始化
        decoder_cfg = dict(
            enable_frame_skip=False,  # 是否跳帧
            output_width=model_w,  # 解码输出宽度（需与检测引擎输入一致）
            output_height=model_h,  # 解码输出高度（需与检测引擎输入一致）
            enable_auto_reconnect=True,  # 是否启用自动重连
            reconnect_delay_ms=2000,  # 重连间隔（毫秒）
            max_reconnects=5,  # 单次重连最大尝试次数，超过后抛出异常结束当前 Pipeline
            open_timeout_ms=5000,  # 打开流超时（毫秒）
            read_timeout_ms=5000,  # 读取数据包超时（毫秒）
            buffer_size=4 * 1024 * 1024,  # 4MB 缓冲区，用于抖动容忍
            max_delay_ms=200,  # 允许的最大解码延迟（毫秒）
            reorder_queue_size=4,  # B 帧重排队列长度
            decoder_threads=1,  # 解码线程数
            surfaces=3,  # 用于缓冲的 CUDA surface 数量
        )
        decoder = Decoder(self.input_url, **decoder_cfg)
        if decoder_cfg["enable_frame_skip"]:
            dec_fps = decoder.get_fps() / 2
        else:
            dec_fps = decoder.get_fps()

        # 编码初始化
        encoder = Encoder(
            output_url=self.output_url,  # 输出 URL 或文件路径
            width=model_det.w,  # 编码输出宽度
            height=model_det.h,  # 编码输出高度
            codec="libx264",  # 编码使用的视频编码器
            bitrate=1000000,  # 编码目标码率（bps）
            fps=dec_fps,  # 实际处理帧率，用于编码器码率控制
        )

        # 跟踪初始化
        tracker = FastByteTracker(
            frame_rate=dec_fps,
            track_thresh=0.5,  # 高置信度检测阈值
            track_buffer=30,  # 丢失目标容忍帧数
            match_thresh=0.8,  # 一级匹配 IoU 阈值
            mot20=False,
        )

        frame_count = 0
        sum_wait = 0
        sum_det = 0
        sum_track = 0
        sum_encode = 0

        while 1:
            t0 = time.time()

            # 解码
            # pts 为帧时间戳
            try:
                frame, pts = decoder.next_frame()
            except Exception as e:
                print(f"{self.input_url} 读取结束: {e}")
                break

            frame_count += 1

            t1 = time.time()

            # 推理
            det_results = model_det(frame)  # (N, 6): x1,y1,x2,y2,score,cls

            t2 = time.time()

            # 跟踪
            det_results = det_results.cpu().numpy()
            tracker_results = tracker.update(det_results)  # (N, 7): x1,y1,x2,y2,score,cls,track_id

            t3 = time.time()

            # 绘制
            annotated_frame = frame.cpu().numpy()

            for x1, y1, x2, y2, score, cls_id, track_id in tracker_results:
                osd.draw_bbox(
                    annotated_frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    color=(64, 64, 255),
                    label=f"#{int(track_id)} {int(cls_id)}",
                )

            encoder.encode(torch.from_numpy(annotated_frame), pts)

            t4 = time.time()

            sum_wait += t1 - t0
            sum_det += t2 - t1
            sum_track += t3 - t2
            sum_encode += t4 - t3

            if frame_count == 1000:
                print(
                    f"[{time.strftime('%m/%d/%Y-%H:%M:%S', time.localtime())}] {self.input_url}, "
                    f"Det: {sum_det:.2f}ms, "
                    f"Track: {sum_track:.2f}ms, "
                    f"Encode: {sum_encode:.2f}ms, "
                    f"Wait: {sum_wait:.2f}ms "
                )
                frame_count = 0
                sum_det = 0
                sum_track = 0
                sum_encode = 0
                sum_wait = 0

        # 编码结束
        encoder.finish()


if __name__ == "__main__":

    args = [
        {
            "gpu": 0,
            "engine_path": "./yolo26n.engine",
            "input_url": "rtsp://172.16.3.210:8554/live/stream1",
            "output_url": "rtmp://172.16.3.194:1935/live/out0",
        },
        {
            "gpu": 0,
            "engine_path": "./yolo26n.engine",
            "input_url": "rtsp://172.16.3.210:8554/live/stream1",
            "output_url": "rtmp://172.16.3.194:1935/live/out1",
        },
        {
            "gpu": 0,
            "engine_path": "./yolo26n.engine",
            "input_url": "rtsp://172.16.3.210:8554/live/stream1",
            "output_url": "rtmp://172.16.3.194:1935/live/out2",
        },
        {
            "gpu": 0,
            "engine_path": "./yolo26n.engine",
            "input_url": "rtsp://172.16.3.210:8554/live/stream1",
            "output_url": "rtmp://172.16.3.194:1935/live/out3",
        },
        {
            "gpu": 0,
            "engine_path": "./yolo26n.engine",
            "input_url": "rtsp://172.16.3.210:8554/live/stream1",
            "output_url": "rtmp://172.16.3.194:1935/live/out4",
        },
        {
            "gpu": 0,
            "engine_path": "./yolo26n.engine",
            "input_url": "rtsp://172.16.3.210:8554/live/stream1",
            "output_url": "rtmp://172.16.3.194:1935/live/out5",
        },
        {
            "gpu": 0,
            "engine_path": "./yolo26n.engine",
            "input_url": "rtsp://172.16.3.210:8554/live/stream1",
            "output_url": "rtmp://172.16.3.194:1935/live/out6",
        },
        {
            "gpu": 0,
            "engine_path": "./yolo26n.engine",
            "input_url": "rtsp://172.16.3.210:8554/live/stream1",
            "output_url": "rtmp://172.16.3.194:1935/live/out7",
        },
        {
            "gpu": 0,
            "engine_path": "./yolo26n.engine",
            "input_url": "rtsp://172.16.3.210:8554/live/stream1",
            "output_url": "rtmp://172.16.3.194:1935/live/out8",
        },
        {
            "gpu": 0,
            "engine_path": "./yolo26n.engine",
            "input_url": "rtsp://172.16.3.210:8554/live/stream1",
            "output_url": "rtmp://172.16.3.194:1935/live/out9",
        },
        {
            "gpu": 0,
            "engine_path": "./yolo26n.engine",
            "input_url": "rtsp://172.16.3.210:8554/live/stream1",
            "output_url": "rtmp://172.16.3.194:1935/live/out10",
        },
        {
            "gpu": 0,
            "engine_path": "./yolo26n.engine",
            "input_url": "rtsp://172.16.3.210:8554/live/stream1",
            "output_url": "rtmp://172.16.3.194:1935/live/out11",
        },
        {
            "gpu": 0,
            "engine_path": "./yolo26n.engine",
            "input_url": "rtsp://172.16.3.210:8554/live/stream1",
            "output_url": "rtmp://172.16.3.194:1935/live/out12",
        },
        {
            "gpu": 0,
            "engine_path": "./yolo26n.engine",
            "input_url": "rtsp://172.16.3.210:8554/live/stream1",
            "output_url": "rtmp://172.16.3.194:1935/live/out13",
        },
        {
            "gpu": 0,
            "engine_path": "./yolo26n.engine",
            "input_url": "rtsp://172.16.3.210:8554/live/stream1",
            "output_url": "rtmp://172.16.3.194:1935/live/out14",
        },
        {
            "gpu": 0,
            "engine_path": "./yolo26n.engine",
            "input_url": "rtsp://172.16.3.210:8554/live/stream1",
            "output_url": "rtmp://172.16.3.194:1935/live/out15",
        },
    ]

    # 使用 'spawn' 启动方式可避免 CUDA 上下文继承问题，
    # 确保每个子进程独立初始化 CUDA。
    # 配合 NVIDIA MPS（多进程服务）时，spawn 模式可让
    # 多个进程共享同一块 GPU 的计算资源，提升并发效率。
    mp.set_start_method("spawn")
    process_pool = []

    for i in args:
        p = Pipeline(**i)
        p.start()
        process_pool.append(p)

        time.sleep(3)  # 避免同时建立大量 RTSP 连接冲垮网络

    for p in process_pool:
        p.join()
