#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <cuda_runtime.h>

#include "eglstreamConsumer.hpp"
#include "EGLAPIAccessors.hpp"

EGLStreamConsumer::EGLStreamConsumer() : cudaConnection(NULL), m_stream(NULL), m_resource(0)
{
    // display = EGL_NO_DISPLAY;
    // stream = EGL_NO_STREAM_KHR;

    // this->fifoLength = fifoLength;
    // if (fifoLength > 0) {
    //     fifoMode = true;
    // } else {
    //     fifoMode = false;
    // }
    // this->latency = latency;
    // this->width = width;
    // this->height = height;

    // printf("CUDA producer initializing EGL display.\n");
    // if (!initEGLDisplay()) {
    //     printf("Cannot initialize EGL display.\n");
    //     return;
    // }

    // printf("CUDA producer initializing EGL stream.\n");
    // if (!initEGLStream()) {
    //     printf("Cannot initialize EGL Stream.\n");
    //     return;
    // }
}

EGLStreamConsumer::~EGLStreamConsumer()
{
    // finalizeEGLCudaProducer();
    // finalizeEGLStream();
}

bool EGLStreamConsumer::connectEGLProducer(EGLStreamKHR stream)
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

bool EGLStreamConsumer::acquireFrame()
{

    CUresult r = cuEGLStreamConsumerAcquireFrame(&cudaConnection, &m_resource, &m_stream, -1);

    if (r == CUDA_SUCCESS)
    {
        cuGraphicsResourceGetMappedEglFrame(&m_frame, m_resource, 0, 0);

        if (m_resource)
        {
            cuEGLStreamConsumerReleaseFrame(&cudaConnection, m_resource, &m_stream);

            std::cout << "Succesfully released" << std::endl;

            return true;
        }
    }

    return false;

    // printf("Connect EGL stream to cuda producer.\n");

    // if (cudaFree(nullptr) != cudaSuccess) {
    //     printf("Failed to initialize CUDA context.\n");
    //     return false;
    // }

    // CUresult ret = cuEGLStreamConsumerConnect(&cudaConnection, stream);
    // if (ret != CUDA_SUCCESS) {
    //     printf("Connect CUDA producer ERROR %d.\n", ret);
    //     return false;
    // }
}

// int EGLStreamConsumer::presentFrameBuffers(int bufferNum)
// {
//     CUresult ret;

//     if (cudaFree(nullptr) != cudaSuccess) {
//         printf("Failed to initialize CUDA context.\n");
//         return -1;
//     }

//     for (int i = 0; i < bufferNum; i++) {
//         CUarray cudaArr[3] = {0};
//         CUDA_ARRAY3D_DESCRIPTOR desc = {0};
//         desc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
//         desc.Depth = 1;
//         desc.NumChannels = 1;
//         desc.Flags = CUDA_ARRAY3D_SURFACE_LDST;
//         for (int i = 0; i < 3; i++) {
//             if (i == 0) {
//                 desc.Width = width;
//                 desc.Height = height;
//             } else {
//                 desc.Width = width / 2;
//                 desc.Height = height / 2;
//             }

//             ret = cuArray3DCreate(&cudaArr[i], &desc);
//             if (ret != CUDA_SUCCESS) {
//                 printf("CUDA create 3D array failed: %d.\n", ret);
//                 return -1;
//             }
//         }

//         CUeglFrame eglFrame;
//         eglFrame.planeCount = 3;
//         eglFrame.numChannels = 1;
//         eglFrame.width = width;
//         eglFrame.height = height;
//         eglFrame.depth = 1;
//         eglFrame.pitch = 0;
//         eglFrame.cuFormat = CU_AD_FORMAT_UNSIGNED_INT8;
//         eglFrame.eglColorFormat = CU_EGL_COLOR_FORMAT_YUV420_PLANAR;
//         eglFrame.frameType = CU_EGL_FRAME_TYPE_ARRAY;
//         eglFrame.frame.pArray[0] = cudaArr[0];
//         eglFrame.frame.pArray[1] = cudaArr[1];
//         eglFrame.frame.pArray[2] = cudaArr[2];

//         printf("CUDA producer present frame: %p.\n", eglFrame.frame.pArray[0]);

//         CUresult ret = cuEGLStreamProducerPresentFrame(&cudaConnection, eglFrame, nullptr);
//         if (ret != CUDA_SUCCESS) {
//             printf("CUDA producer present frame failed: %d.\n", ret);
//             return -1;
//         }
//     }

//     return 0;
// }

// int EGLStreamConsumer::presentFrame(CUdeviceptr data)
// {
//     CUresult ret;

//     if (cudaFree(nullptr) != cudaSuccess) {
//         printf("Failed to initialize CUDA context.\n");
//         return -1;
//     }

//     CUeglFrame eglFrame;
//     ret = cuEGLStreamProducerReturnFrame(&cudaConnection, &eglFrame, nullptr);
//     if (ret != CUDA_SUCCESS) {
//         printf("CUDA producer return frame failed: %d.\n", ret);
//         return -1;
//     }

//     printf("CUDA producer return frame: %p.\n", eglFrame.frame.pArray[0]);

//     CUDA_MEMCPY3D cpdesc;
//     size_t offsets[3], copyWidth[3], copyHeight[3];
//     offsets[0] = 0;
//     offsets[1] = width * height;
//     offsets[2] = offsets[1] + width * height / 4;
//     copyWidth[0] = width;
//     copyWidth[1] = width / 2;
//     copyWidth[2] = width / 2;
//     copyHeight[0] = height;
//     copyHeight[1] = height / 2;
//     copyHeight[2] = height / 2;

//     for (int i = 0; i < 3; i++) {
//         memset(&cpdesc, 0, sizeof(cpdesc));
//         cpdesc.srcMemoryType = CU_MEMORYTYPE_DEVICE;
//         cpdesc.srcDevice = (CUdeviceptr)((char *)data + offsets[i]);
//         cpdesc.dstMemoryType = CU_MEMORYTYPE_ARRAY;
//         cpdesc.dstArray = eglFrame.frame.pArray[i];
//         cpdesc.WidthInBytes = copyWidth[i];
//         cpdesc.Height = copyHeight[i];
//         cpdesc.Depth = 1;

//         ret = cuMemcpy3D(&cpdesc);
// //        ret = cuMemcpyDtoA(eglFrame.frame.pArray[i], 0, (CUdeviceptr)((char *)data + offsets[i]), 1);
//         if (ret != CUDA_SUCCESS) {
//             printf("CUDA producer copy data to EGL frame failed: %d.\n", ret);
//             return -1;
//         }
//     }

//     ret = cuEGLStreamProducerPresentFrame(&cudaConnection, eglFrame, nullptr);
//     if (ret != CUDA_SUCCESS) {
//         printf("CUDA producer present frame failed: %d.\n", ret);
//         return -1;
//     }

//     return 0;
// }

// bool EGLStreamConsumer::initEGLDisplay()
// {
//     // Obtain the EGL display
//     display = EGLDisplayAccessor::getInstance();
//     if (display == EGL_NO_DISPLAY) {
//         printf("Obtain EGL display failed.\n");
//         return false;
//     }

//     return true;
// }

// bool EGLStreamConsumer::initEGLStream()
// {
//     const EGLint streamAttrMailboxMode[] = { EGL_NONE };
//     const EGLint streamAttrFIFOMode[] = { EGL_STREAM_FIFO_LENGTH_KHR, fifoLength, EGL_NONE };

//     if (!setupEGLExtensions()) {
//         return false;
//     }

//     stream = eglCreateStreamKHR(display, fifoMode ? streamAttrFIFOMode : streamAttrMailboxMode);
//     if (stream == EGL_NO_STREAM_KHR) {
//         printf("Couldn't create stream.\n");
//         return false;
//     }

//     if (!eglStreamAttribKHR(display, stream, EGL_CONSUMER_LATENCY_USEC_KHR, latency)) {
//         printf("Producer: streamAttribKHR EGL_CONSUMER_LATENCY_USEC_KHR failed.\n");
//     }
//     if (!eglStreamAttribKHR(display, stream, EGL_CONSUMER_ACQUIRE_TIMEOUT_USEC_KHR, latency)) {
//         printf("Producer: streamAttribKHR EGL_CONSUMER_ACQUIRE_TIMEOUT_USEC_KHR failed.\n");
//     }

//     // Get stream attributes
//     if (!eglQueryStreamKHR(display, stream, EGL_STREAM_FIFO_LENGTH_KHR, &fifoLength)) {
//         printf("Producer: eglQueryStreamKHR EGL_STREAM_FIFO_LENGTH_KHR failed.\n");
//     }
//     if (!eglQueryStreamKHR(display, stream, EGL_CONSUMER_LATENCY_USEC_KHR, &latency)) {
//         printf("Producer: eglQueryStreamKHR EGL_CONSUMER_LATENCY_USEC_KHR failed.\n");
//     }

//     if (fifoMode != (fifoLength > 0)) {
//         printf("EGL Stream consumer - Unable to set FIFO mode.\n");
//         fifoMode = false;
//     }
//     if (fifoMode) {
//         printf("EGL Stream consumer - Mode: FIFO, Length: %d, latency %d.\n", fifoLength, latency);
//     } else {
//         printf("EGL Stream consumer - Mode: Mailbox.\n");
//     }

//     return true;
// }

// void EGLStreamConsumer::finalizeEGLStream()
// {
//     if (stream != EGL_NO_STREAM_KHR) {
//         eglDestroyStreamKHR(display, stream);
//         stream = EGL_NO_STREAM_KHR;
//     }
// }

// void EGLStreamConsumer::finalizeEGLCudaProducer()
// {
//     if (cudaConnection) {
//         if (cudaFree(nullptr) != cudaSuccess) {
//             printf("Failed to initialize CUDA context.\n");
//             return;
//         }

//         cuEGLStreamProducerDisconnect(&cudaConnection);
//         cudaConnection = nullptr;
//     }
// }