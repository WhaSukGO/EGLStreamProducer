// #ifndef EGLSTREAMPRODUCER_H
// #define EGLSTREAMPRODUCER_H

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <cudaEGL.h>
#include "opencv2/core/cuda.hpp"

#include <npp.h>


class EGLStreamConsumer
{
public:
    EGLStreamConsumer(int fifoLength, int latency, int width, int height);
    ~EGLStreamConsumer();

    EGLDisplay getEGLDisplay() {
        return display;
    }

    EGLStreamKHR getEGLStream() {
        return stream;
    }

    cv::cuda::GpuMat acquireFrame();

private:
    // EGLStream to consumer
    CUeglStreamConnection cudaConnection;
    CUstream m_stream;
    CUgraphicsResource m_resource;
    CUeglFrame m_frame;

    // GStreamer to EGLStream
    EGLDisplay display;
    EGLStreamKHR stream;
    int fifoLength;
    bool fifoMode;
    int latency;
    int width;
    int height;

    cudaError_t cudaErr;
    NppStatus nppErr;

    uchar* d_ABGR;

    bool initEGLDisplay();
    bool initEGLStream();
    bool initEGLCudaConsumer();
    void finalizeEGLStream();
    void finalizeEGLCudaProducer();
};


// #endif // EGLSTREAMPRODUCER_H