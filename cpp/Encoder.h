#pragma once

#include <torch/extension.h>

#include <cstdint>
#include <string>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
}

class Encoder {
public:
    // 输出时间戳与帧时长完全由 encode() 传入的 pts 决定；fps 只用于编码器内部的码率控制与时序参数
    Encoder(const std::string& filename, int width, int height, std::string codec, int bitrate = 2000000,
            double fps = 25.0);
    ~Encoder();

    void encode(torch::Tensor frame, double pts);
    void finish();

private:
    void init_ffmpeg(const std::string& codec);
    void cleanup();

    std::string filename;
    int         width;
    int         height;
    int         bitrate;
    double      fps;

    AVFormatContext* format_ctx   = nullptr;
    AVCodecContext*  codec_ctx    = nullptr;
    AVStream*        video_stream = nullptr;

    AVFrame*  frame   = nullptr;  // 双缓冲 A：编码器内部会持有上一帧，交替写入可避免整帧拷贝
    AVFrame*  frame_b = nullptr;  // 双缓冲 B
    AVPacket* packet  = nullptr;
    bool      use_b   = false;

    int64_t time_base_den  = 90000;  // 输出时间基分母（pts 驱动，{1, time_base_den}）
    int64_t frame_duration = 0;      // 由相邻两帧 pts 差值得到，供封装层计算时长
    int64_t last_pts       = -1;     // 已写入编码器的最后一个 pts
    int64_t frame_index    = 0;
    bool    is_finished    = false;

    struct SwsContext* sws_ctx = nullptr;
};
