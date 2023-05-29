// #ifndef EGLSTREAMPRODUCER_H
// #define EGLSTREAMPRODUCER_H

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <cudaEGL.h>

class EGLStreamConsumer
{
public:
    EGLStreamConsumer();
    ~EGLStreamConsumer();

    // EGLDisplay getEGLDisplay() {
    //     return display;
    // }

    // EGLStreamKHR getEGLStream() {
    //     return stream;
    // }

    bool connectEGLProducer(EGLStreamKHR stream);
    bool acquireFrame();

private:
    CUeglStreamConnection cudaConnection;
    CUstream m_stream;
    CUgraphicsResource m_resource;
    CUeglFrame m_frame;
};


// #endif // EGLSTREAMPRODUCER_H