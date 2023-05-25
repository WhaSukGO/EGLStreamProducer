#ifndef EGLSTREAMPRODUCER_H
#define EGLSTREAMPRODUCER_H

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <cudaEGL.h>

class EGLStreamProducer
{
public:
    EGLStreamProducer(int fifoLength, int latency, int width, int height);
    ~EGLStreamProducer();

    EGLDisplay getEGLDisplay() {
        return display;
    }

    EGLStreamKHR getEGLStream() {
        return stream;
    }

    bool connectEGLProducer();
    int presentFrameBuffers(int bufferNum);
    int presentFrame(CUdeviceptr data);

private:
    bool initEGLDisplay();
    bool initEGLStream();
    void finalizeEGLStream();
    void finalizeEGLCudaProducer();

    EGLDisplay display;
    EGLStreamKHR stream;
    int fifoLength;
    bool fifoMode;
    int latency;
    int width;
    int height;

    CUeglStreamConnection cudaConnection;
};


#endif // EGLSTREAMPRODUCER_H