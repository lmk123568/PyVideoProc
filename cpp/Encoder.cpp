#include "Encoder.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

extern "C" {
#include <libavutil/error.h>
#include <libavutil/rational.h>
#include <libswscale/swscale.h>
}

namespace {

std::string ff_err(int ret) {
    char buf[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(ret, buf, sizeof(buf));
    return std::string(buf);
}

// 输出时间基：90000 是 MP4 的原生时间基，pts 换算到该时间基可保持亚毫秒精度
constexpr int64_t kDefaultTimeBaseDen = 90000;
// FLV/RTMP 时间戳以毫秒为单位
constexpr int64_t kFlvTimeBaseDen = 1000;
// 编码线程数上限：限制 CPU 占用，避免与解码/推理抢核
constexpr int kThreadCount = 4;

}  // namespace

Encoder::Encoder(const std::string& filename, int width, int height, std::string codec, int bitrate, double fps)
    : filename(filename), width(width), height(height), bitrate(bitrate), fps(fps) {
    try {
        init_ffmpeg(codec);
    } catch (...) {
        // 构造失败时析构函数不会执行，需要手动释放已分配资源
        cleanup();
        throw;
    }
}

Encoder::~Encoder() {
    finish();
    cleanup();
}

void Encoder::init_ffmpeg(const std::string& codec) {
    av_log_set_level(AV_LOG_ERROR);

    cleanup();
    is_finished = false;

    const char* format_name = nullptr;
    if (filename.find("rtsp://") == 0) {
        format_name = "rtsp";
    } else if (filename.find("rtmp://") == 0) {
        format_name = "flv";
    }

    int ret = avformat_alloc_output_context2(&format_ctx, nullptr, format_name, filename.c_str());
    if (!format_ctx) {
        ret = avformat_alloc_output_context2(&format_ctx, nullptr, "flv", filename.c_str());
    }
    if (!format_ctx) {
        throw std::runtime_error("Could not create output context for " + filename);
    }

    // 默认且仅支持 CPU 编码（libx264）；codec 参数保留用于接口兼容
    (void)codec;
    const AVCodec* encoder_codec = avcodec_find_encoder_by_name("libx264");
    if (!encoder_codec) {
        throw std::runtime_error("libx264 codec not found");
    }

    video_stream = avformat_new_stream(format_ctx, encoder_codec);
    if (!video_stream) {
        throw std::runtime_error("Could not allocate stream");
    }

    codec_ctx = avcodec_alloc_context3(encoder_codec);
    if (!codec_ctx) {
        throw std::runtime_error("Could not allocate codec context");
    }

    // 时间基只用于 pts 换算，帧间隔由每帧传入的 pts 决定
    time_base_den  = (format_name && std::string(format_name) == "flv") ? kFlvTimeBaseDen : kDefaultTimeBaseDen;

    codec_ctx->width        = width;
    codec_ctx->height       = height;
    codec_ctx->time_base    = {1, static_cast<int>(time_base_den)};
    // 必须显式设置帧率：libx264 在 framerate 未设置时会退化成用 time_base 当帧率（1/90000 → 90000fps），
    // 码率控制按帧分到的比特数近似为 0，量化参数打满，重建帧被压平为纯灰（Y=U=V=128）。
    codec_ctx->framerate = av_d2q(fps > 0 ? fps : 25.0, 1000000);
    codec_ctx->bit_rate     = bitrate;
    // I 帧间隔固定为 50 帧：I 帧编码成本远高于 P 帧，gop 越大越省（代价是随机访问/抗丢包变差）
    codec_ctx->gop_size     = 50;
    codec_ctx->max_b_frames = 0;
    codec_ctx->pix_fmt      = AV_PIX_FMT_YUV420P;
    // 显式限制编码线程数，避免 libx264 默认吃满所有核；片内切片线程更适合低延迟
    codec_ctx->thread_count = kThreadCount;
    codec_ctx->thread_type  = FF_THREAD_SLICE;

    video_stream->time_base = codec_ctx->time_base;

    if (format_ctx->oformat->flags & AVFMT_GLOBALHEADER) {
        codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    AVDictionary* opts = NULL;
    // ultrafast 相比 veryfast 约省一半以上编码耗时；tune zerolatency 已禁用 B 帧与预分析
    av_dict_set(&opts, "preset", "ultrafast", 0);
    av_dict_set(&opts, "tune", "zerolatency", 0);
    av_dict_set(&opts, "profile", "baseline", 0);
    // 显式锁死省算参数，避免后续修改 preset 时回退
    av_dict_set(&opts, "x264-params", "rc-lookahead=0:sync-lookahead=0:bframes=0:ref=1", 0);

    ret = avcodec_open2(codec_ctx, encoder_codec, &opts);
    av_dict_free(&opts);
    if (ret < 0) {
        throw std::runtime_error("Could not open codec libx264: " + ff_err(ret));
    }

    ret = avcodec_parameters_from_context(video_stream->codecpar, codec_ctx);
    if (ret < 0) {
        throw std::runtime_error("Could not copy stream parameters: " + ff_err(ret));
    }

    if (!(format_ctx->oformat->flags & AVFMT_NOFILE)) {
        ret = avio_open(&format_ctx->pb, filename.c_str(), AVIO_FLAG_WRITE);
        if (ret < 0) {
            throw std::runtime_error("Could not open output: " + filename + ": " + ff_err(ret));
        }
    }

    ret = avformat_write_header(format_ctx, nullptr);
    if (ret < 0) {
        throw std::runtime_error("Error occurred when opening output: " + ff_err(ret));
    }

    frame   = av_frame_alloc();
    frame_b = av_frame_alloc();
    packet  = av_packet_alloc();
    if (!frame || !frame_b || !packet) {
        throw std::runtime_error("Failed to allocate frame/packet");
    }

    for (AVFrame* f : {frame, frame_b}) {
        f->format = codec_ctx->pix_fmt;
        f->width  = width;
        f->height = height;
        ret       = av_frame_get_buffer(f, 32);
        if (ret < 0) {
            throw std::runtime_error("Failed to allocate SW frame buffer: " + ff_err(ret));
        }
    }

    // 这里只做 BGR24→YUV420P 颜色转换（libx264 要求 YUV420P），尺寸不变，没有缩放
    sws_ctx = sws_getContext(
        width, height, AV_PIX_FMT_BGR24, width, height, codec_ctx->pix_fmt, SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);
    if (!sws_ctx) {
        throw std::runtime_error("Failed to create sws context");
    }

    std::cerr << "[Encoder] 使用 CPU 编码(libx264, preset=ultrafast, threads=" << kThreadCount
              << ", gop=" << codec_ctx->gop_size << ") 输出到 " << filename << ", time_base=1/" << time_base_den
              << std::endl;
}

void Encoder::cleanup() {
    if (frame) av_frame_free(&frame);
    if (frame_b) av_frame_free(&frame_b);
    if (packet) av_packet_free(&packet);
    if (codec_ctx) avcodec_free_context(&codec_ctx);
    if (format_ctx) {
        if (!(format_ctx->oformat->flags & AVFMT_NOFILE)) {
            avio_closep(&format_ctx->pb);
        }
        avformat_free_context(format_ctx);
    }
    if (sws_ctx) sws_freeContext(sws_ctx);

    frame          = nullptr;
    frame_b        = nullptr;
    packet         = nullptr;
    codec_ctx      = nullptr;
    format_ctx     = nullptr;
    video_stream   = nullptr;
    sws_ctx        = nullptr;
    use_b          = false;
    last_pts       = -1;
    frame_index    = 0;
    time_base_den  = kDefaultTimeBaseDen;
    frame_duration = 0;
}

void Encoder::encode(torch::Tensor tensor, double pts) {
    if (!codec_ctx || !frame || !frame_b) {
        throw std::runtime_error("Encoder is not initialized");
    }
    if (is_finished) {
        throw std::runtime_error("Encoder is already finished");
    }
    if (tensor.dtype() != torch::kUInt8) {
        throw std::runtime_error("Input tensor must be uint8 for libx264");
    }
    if (tensor.is_cuda()) {
        tensor = tensor.cpu();
    }
    if (tensor.dim() != 3 || tensor.size(0) != height || tensor.size(1) != width || tensor.size(2) != 3) {
        throw std::runtime_error("Input tensor must be HWC uint8 BGR with encoder resolution");
    }
    if (!tensor.is_contiguous()) {
        tensor = tensor.contiguous();
    }

    // 双缓冲：编码器会持有上一帧的引用，交替写入可让 av_frame_make_writable 变成空操作
    AVFrame* out = use_b ? frame_b : frame;
    use_b        = !use_b;

    int ret = av_frame_make_writable(out);
    if (ret < 0) {
        throw std::runtime_error("Frame not writable: " + ff_err(ret));
    }

    uint8_t* src_slices[1]  = {tensor.data_ptr<uint8_t>()};
    int      src_strides[1] = {width * 3};

    int scaled = sws_scale(sws_ctx, src_slices, src_strides, 0, height, out->data, out->linesize);
    if (scaled != height) {
        throw std::runtime_error("sws_scale failed");
    }

    // 输出时间戳与帧时长完全由 pts（秒）决定
    int64_t tb_pts = static_cast<int64_t>(std::llround(pts * static_cast<double>(time_base_den)));
    if (last_pts >= 0) {
        frame_duration = tb_pts - last_pts;  // 帧时长 = 相邻两帧 pts 差值
        if (frame_duration <= 0) {
            tb_pts         = last_pts + 1;  // 保证 pts 严格递增，避免 0 时长帧
            frame_duration = 1;
        }
    } else {
        frame_duration = 0;  // 首帧时长未知，待下一帧 pts 到达后确定
    }
    last_pts      = tb_pts;
    out->pts      = tb_pts;
    out->duration = frame_duration;  // 供封装层计算音频/视频时长
    ++frame_index;

    ret = avcodec_send_frame(codec_ctx, out);
    if (ret < 0) {
        throw std::runtime_error("Error sending frame to encoder: " + ff_err(ret));
    }
    while (ret >= 0) {
        ret = avcodec_receive_packet(codec_ctx, packet);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break;
        } else if (ret < 0) {
            throw std::runtime_error("Error encoding frame: " + ff_err(ret));
        }

        packet->duration = frame_duration;
        av_packet_rescale_ts(packet, codec_ctx->time_base, video_stream->time_base);
        packet->stream_index = video_stream->index;

        ret = av_write_frame(format_ctx, packet);
        av_packet_unref(packet);
        if (ret < 0) {
            throw std::runtime_error("Error writing packet: " + ff_err(ret));
        }
    }
}

void Encoder::finish() {
    if (is_finished) return;
    is_finished = true;

    if (!codec_ctx || !format_ctx || !packet) return;

    int ret = avcodec_send_frame(codec_ctx, nullptr);

    while (true) {
        ret = avcodec_receive_packet(codec_ctx, packet);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break;
        } else if (ret < 0) {
            std::cerr << "Error flushing encoder: " << ff_err(ret) << std::endl;
            break;
        }

        packet->duration = frame_duration;
        av_packet_rescale_ts(packet, codec_ctx->time_base, video_stream->time_base);
        packet->stream_index = video_stream->index;

        av_write_frame(format_ctx, packet);
        av_packet_unref(packet);
    }

    av_write_trailer(format_ctx);
}
