#pragma once

#include <torch/extension.h>

#include <memory>
#include <string>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
}

class Encoder {
public:
    Encoder(const std::string& filename, int width, int height, int fps, std::string codec, int bitrate = 2000000);
    ~Encoder();

    void encode(torch::Tensor frame, double pts = -1.0);
    void finish();

private:
    void init_ffmpeg(std::string codec);
    void cleanup();
    void process_frame(torch::Tensor frame);

    std::string filename;
    int         width;
    int         height;
    int         fps;
    int         bitrate;

    AVFormatContext* format_ctx    = nullptr;
    AVCodecContext*  codec_ctx     = nullptr;
    AVBufferRef*     hw_device_ctx = nullptr;
    AVStream*        video_stream  = nullptr;

    AVFrame*  frame  = nullptr;
    AVPacket* packet = nullptr;

    int64_t frame_index = 0;
    bool    is_finished = false;

    // For NPP/CUDA conversion
    // We might need intermediate buffers if NPP requires it,
    // but usually we can map pointers directly if strides are aligned.
};
