FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel

RUN \
    sed -i "s/archive.ubuntu./mirrors.aliyun./g" /etc/apt/sources.list && \
    sed -i "s/deb.debian.org/mirrors.aliyun.com/g" /etc/apt/sources.list && \
    sed -i "s/security.debian.org\/debian-security/mirrors.aliyun.com\/debian-security/g" /etc/apt/sources.list && \
    sed -i "s/httpredir.debian.org\/debian-security/mirrors.aliyun.com\/debian-security/g" /etc/apt/sources.list && \
    apt update -y && \
    apt install -y --no-install-recommends wget curl ffmpeg vim pkg-config libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libswresample-dev && \
    pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple ultralytics onnx onnxslim tensorrt && \
    rm -rf /var/lib/apt/lists/*

# docker build -t pynvvideopipe:cuda12.6 .
