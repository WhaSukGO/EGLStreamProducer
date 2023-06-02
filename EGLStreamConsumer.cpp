#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <cuda_runtime.h>

#include "EGLStreamConsumer.hpp"
#include "EGLAPIAccessors.hpp"

#include <npp.h>
#include <nppi_color_conversion.h>

#include "opencv2/highgui.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaimgproc.hpp"

EGLStreamConsumer::EGLStreamConsumer(int fifoLength, int latency, int width, int height) : cudaConnection(NULL), m_stream(NULL), m_resource(0)
{
    display = EGL_NO_DISPLAY;
    stream = EGL_NO_STREAM_KHR;

    this->fifoLength = fifoLength;
    this->latency = latency;
    this->width = width;
    this->height = height;

    if (fifoLength > 0)
    {
        fifoMode = true;
    }
    else
    {
        fifoMode = false;
    }

    if (!initEGLDisplay())
    {
        printf("Cannot initialize EGL display.\n");
        return;
    }

    if (!initEGLStream())
    {
        printf("Cannot initialize EGL Stream.\n");
        return;
    }

    if (!initEGLCudaConsumer())
    {
        printf("Cannot initialize EGLCudaConsumer.\n");
        return;
    }

    if (!cudaMalloc(&d_ABGR, sizeof(uchar) * width * height * 4))
    {
        printf("Failed to allocate a device memory for ABGR.\n");
        return;
    }
}

EGLStreamConsumer::~EGLStreamConsumer()
{
    finalizeEGLCudaProducer();
    finalizeEGLStream();
}

bool EGLStreamConsumer::initEGLCudaConsumer()
{
    printf("Connect EGL stream to cuda producer.\n");

    if (cudaFree(nullptr) != cudaSuccess)
    {
        printf("Failed to initialize CUDA context.\n");
        return false;
    }

    CUresult ret = cuEGLStreamConsumerConnect(&cudaConnection, stream);
    if (ret != CUDA_SUCCESS)
    {
        printf("Connect CUDA producer ERROR %d.\n", ret);
        return false;
    }

    return true;
}

cv::cuda::GpuMat EGLStreamConsumer::acquireFrame()
{
    EGLint streamState = 0;

    if (!eglQueryStreamKHR(display, stream, EGL_STREAM_STATE_KHR, &streamState))
    {
        printf("Cuda consumer, eglQueryStreamKHR EGL_STREAM_STATE_KHR failed.\n");
        return cv::cuda::GpuMat();
    }

    CUresult r = cuEGLStreamConsumerAcquireFrame(&cudaConnection, &m_resource, &m_stream, -1);

    if (r == CUDA_SUCCESS)
    {
        cuGraphicsResourceGetMappedEglFrame(&m_frame, m_resource, 0, 0);

        // printf("height: %d\n", m_frame.height);
        // printf("width: %d\n", m_frame.width);
        // printf("depth: %d\n", m_frame.depth);
        // printf("pitch: %d\n", m_frame.pitch);
        // printf("planeCount: %d\n", m_frame.planeCount);
        // printf("numChannels: %d\n", m_frame.numChannels);
        // // 0: ARRAY | 1 : PITCH => PITCH
        // printf("frameType: : %d\n", m_frame.frameType);
        // printf("eglColorFormat: : %x\n", m_frame.eglColorFormat);
        // printf("Memory: %p\n", m_frame.frame.pArray[0]);
        // printf("Memory: %p\n", m_frame.frame.pArray[1]);
        // printf("Memory: %p\n", m_frame.frame.pPitch);

        if (m_resource)
        {
            int width = m_frame.width;
            int height = m_frame.height;
            int pitch = m_frame.pitch;
            void* ptrABGR = m_frame.frame.pPitch[0];
    
            int stepSizeABGR = width * m_frame.numChannels * sizeof(uchar);
            int stepSizeBGR = width * 3 * sizeof(uchar);

            uchar* ptrBGR;
            cudaMalloc(&ptrBGR, stepSizeBGR * height);

            cv::cuda::GpuMat gpuABGR(height, width, CV_8UC4, ptrABGR, pitch);
            cv::cuda::GpuMat gpuBGR(height, width, CV_8UC3, ptrBGR, stepSizeBGR);
            cv::cuda::cvtColor(gpuABGR, gpuBGR, cv::COLOR_RGBA2BGR);

            cuEGLStreamConsumerReleaseFrame(&cudaConnection, m_resource, &m_stream);

            return gpuBGR;
        }
        else
        {
            printf("cuGraphicsResourceGetMappedEglFrame() failed\n");
            return cv::cuda::GpuMat();
        }
    }
    else
    {
        printf("cuEGLStreamConsumerAcquireFrame() failed\n");
        return cv::cuda::GpuMat();
    }

    return cv::cuda::GpuMat();
}


bool EGLStreamConsumer::initEGLDisplay()
{
    // Obtain the EGL display
    display = EGLDisplayAccessor::getInstance();
    if (display == EGL_NO_DISPLAY)
    {
        printf("Obtain EGL display failed.\n");
        return false;
    }

    return true;
}

bool EGLStreamConsumer::initEGLStream()
{
    const EGLint streamAttrMailboxMode[] = {EGL_NONE};
    const EGLint streamAttrFIFOMode[] = {EGL_STREAM_FIFO_LENGTH_KHR, fifoLength, EGL_NONE};

    if (!setupEGLExtensions())
    {
        return false;
    }

    stream = eglCreateStreamKHR(display, fifoMode ? streamAttrFIFOMode : streamAttrMailboxMode);
    if (stream == EGL_NO_STREAM_KHR)
    {
        printf("Couldn't create stream.\n");
        return false;
    }

    if (!eglStreamAttribKHR(display, stream, EGL_CONSUMER_LATENCY_USEC_KHR, latency))
    {
        printf("Producer: streamAttribKHR EGL_CONSUMER_LATENCY_USEC_KHR failed.\n");
    }
    if (!eglStreamAttribKHR(display, stream, EGL_CONSUMER_ACQUIRE_TIMEOUT_USEC_KHR, latency))
    {
        printf("Producer: streamAttribKHR EGL_CONSUMER_ACQUIRE_TIMEOUT_USEC_KHR failed.\n");
    }

    // Get stream attributes
    if (!eglQueryStreamKHR(display, stream, EGL_STREAM_FIFO_LENGTH_KHR, &fifoLength))
    {
        printf("Producer: eglQueryStreamKHR EGL_STREAM_FIFO_LENGTH_KHR failed.\n");
    }
    if (!eglQueryStreamKHR(display, stream, EGL_CONSUMER_LATENCY_USEC_KHR, &latency))
    {
        printf("Producer: eglQueryStreamKHR EGL_CONSUMER_LATENCY_USEC_KHR failed.\n");
    }

    if (fifoMode != (fifoLength > 0))
    {
        printf("EGL Stream consumer - Unable to set FIFO mode.\n");
        fifoMode = false;
    }
    if (fifoMode)
    {
        printf("EGL Stream consumer - Mode: FIFO, Length: %d, latency %d.\n", fifoLength, latency);
    }
    else
    {
        printf("EGL Stream consumer - Mode: Mailbox.\n");
    }

    return true;
}

void EGLStreamConsumer::finalizeEGLStream()
{
    if (stream != EGL_NO_STREAM_KHR)
    {
        eglDestroyStreamKHR(display, stream);
        stream = EGL_NO_STREAM_KHR;
    }
}

void EGLStreamConsumer::finalizeEGLCudaProducer()
{
    if (cudaConnection)
    {
        if (cudaFree(nullptr) != cudaSuccess)
        {
            printf("Failed to initialize CUDA context.\n");
            return;
        }

        cuEGLStreamProducerDisconnect(&cudaConnection);
        cudaConnection = nullptr;
    }
}