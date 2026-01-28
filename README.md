# PyNvVideoPipe

![License](https://img.shields.io/badge/license-Apache_2.0-blue.svg?style=for-the-badge)
![Nvidia](https://img.shields.io/badge/CUDA-12.6.3-76B900?&logoColor=white&style=for-the-badge)
![OS](https://img.shields.io/badge/OS-Linux-FCC624?&logoColor=white&style=for-the-badge)

High-Performance Video Processing Pipeline in Python, Powered by NVIDIA CUDA

Supports multi-stream, multi-GPU, and multi-model inference

Minimizes memory copies and CPU–GPU data transfers for maximum efficiency

基于 NVIDIA CUDA 的 Python 高性能视频处理流水线实现

支持多路视频流、多 GPU 与多模型推理

最大限度减少显存拷贝和 CPU–GPU 数据传输，提升推理效率

|                                                           | Open Source开源 |      Learning Curve学习成本      | Developer-Friendliness二次开发友好度 |          Performance性能          |
| :-------------------------------------------------------: | :-------------: | :------------------------------: | :----------------------------------: | :-------------------------------: |
| [DeepStream](https://developer.nvidia.com/deepstream-sdk) |        ❌        |               High               |                 Low                  |               High                |
| [VideoPipe](https://github.com/sherlockchou86/VideoPipe)  |        ✅        | medium（requires cpp knowledge） |   Medium（requires cpp knowledge）   |              Medium               |
|                            Our                            |        ✅        |               ≈ 0                |           High +++++++++++           | Medium（with some optimizations） |

### Quick Start

##### 1. 准备运行环境

本项目推荐 Docker 容器运行，首先确保本地环境满足以下三个条件

- Docker >= 24.0.0

- NVIDIA Driver >= 590

- NVIDIA Container Toolkit >= 1.13.0

之后 clone 本项目，生成包含完整开发环境的镜像

```bash
git clone https://github.com/lmk123568/PyNvVideoPipe.git
cd PyNvVideoPipe/docker
docker build -t PyNvVideoPipe:cuda12.6 .
```

镜像生成后，进入容器，不报错即成功

```bash
docker run -it \
  --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /.../{your_path}/PyNvVideoPipe:/workspace \
  PyNvVideoPipe:cuda12.6 \
  bash
```

后续示例代码默认在容器内`/workspace`运行

##### 2. 编译硬件加速库实现

```bash
cd /codec

# Two options, pick one
python setup.py build_ext --inplace  # Debug
python setup.py install  # Release
```

> 不推荐自己本地装环境，如果一定要自己装，请参考 Dockerfile

##### 3. 训练模型权重转换

将通过 [ultralytics](https://github.com/ultralytics/ultralytics) 训练的模型导入到`yolo26`目录下，示例模型为 [yolo26n.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt)

```bash
cd /yolo26
python pt2trt.py  --w yolo26n.pt --fp16
```

🚀🚀🚀 推理尺寸建议固定为`(576,1024)`，可以跳过`letterbox`降低计算开销

##### 4. 运行

修改并理解`main.py`

```bash
cd /workspace
python main.py
```

### Benchmark

**Date**: 2026-01-25

**Hardware**: AMD Ryzen 9 5950 X + NVIDIA GeForce RTX 3090

**Test Configuration**: 4 × RTSP Decoders → YOLO26 (TensorRT) → 4 × RTMP Encoders

|                           | CPU     | RAM     | GPU VRAM | **GPU-Util** |
| ------------------------- | ------- | ------- | -------- | ------------ |
| VidepPipe（ffmpeg codec） | 511.6 % | 1.5 GiB | 2677 MiB | 16 %         |
| Our                       | 9.9%    | 1.2GiB  | 3932 MiB | 9%           |

### Notes

- 更多细节和技巧请阅读 `main.py` 注释
- 大简之道是最美的艺术，没有之一
- 工程不是追求完美的数学解，而是在资源受限、时间紧迫、需求模糊的情况下，寻找一个可用的最优解

### License

[Apache 2.0](https://github.com/lmk123568/PyNvVideoPipe/blob/main/LICENSE)

