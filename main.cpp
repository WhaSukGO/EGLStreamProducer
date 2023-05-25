#include <thread>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <gst/gst.h>
#include <cuda_runtime.h>

#include "eglstreamproducer.h"

static const int FrameWidth = 800;
static const int FrameHeight = 600;

static EGLStreamProducer *eglStreamProducer = nullptr;

void producerThreadFunc()
{
    if (cudaFree(nullptr) != cudaSuccess) {
        printf("Failed to initialize CUDA context.\n");
        return;
    }

    CUdeviceptr buffer;
    CUresult ret = cuMemAlloc(&buffer, FrameWidth * FrameHeight * 3 / 2);
    if (ret != CUDA_SUCCESS) {
        g_print("cuMemAlloc failed: %d\n.", ret);
        return;
    }

    int cnt = 0;
    while (cnt < 50) {
        std::this_thread::sleep_for(std::chrono::milliseconds(40));

        cnt++;
        g_print("Present a new frame %d.\n", cnt);
        // call cuEGLStreamProducerReturnFrame to get the returned frame from EGL stream,
        // and then call cuEGLStreamProducerPresentFrame to push the frame back to the EGL stream FIFO.
        eglStreamProducer->presentFrame(buffer);
    }

    cuMemFree(buffer);
}

int main(int argc, char *argv[])
{
    gst_init(nullptr, nullptr);

    GstElement *pipeline = gst_pipeline_new("play");
    if (pipeline == nullptr) {
        g_print("Create pipeline failed.\n");
        return -1;
    }

    GstElement *source = gst_element_factory_make("nveglstreamsrc", nullptr);
    if (source == nullptr) {
        g_print("Create eglstream source failed.\n");
        return -1;
    }

    eglStreamProducer = new EGLStreamProducer(4, 0, FrameWidth, FrameHeight);
    g_object_set(source, "display", eglStreamProducer->getEGLDisplay(), nullptr);
    g_object_set(source, "eglstream", eglStreamProducer->getEGLStream(), nullptr);

    GstElement *capFilter = gst_element_factory_make("capsfilter", nullptr);
    if (capFilter == nullptr) {
        g_print("Create capsfilter failed.\n");
        return -1;
    }

    GstCaps *caps = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "NV12",
                                        "width", G_TYPE_INT, FrameWidth,
                                        "height", G_TYPE_INT, FrameHeight,
                                        "framerate", GST_TYPE_FRACTION, 25, 1, NULL);

    GstCapsFeatures *feature = gst_caps_features_new("memory:NVMM", NULL);
    gst_caps_set_features(caps, 0, feature);

    /* Set capture caps on capture filter */
    g_object_set(capFilter, "caps", caps, NULL);
    gst_caps_unref(caps);

    GstElement *sink = gst_element_factory_make("fakesink", nullptr);
    if (sink == nullptr) {
        g_print("Create overlay sink failed.\n");
        return -1;
    }

    gst_bin_add_many(GST_BIN(pipeline), source, capFilter, sink, nullptr);
    if (!gst_element_link_many(source, capFilter, sink, nullptr)) {
        g_print("Link elememt eglstream source <-> overlay sink failed.\n");
        return -1;
    }

    GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        g_print("Change pipeline state to %s failed.\n", gst_element_state_get_name(GST_STATE_PLAYING));
        return -1;
    }

    if (!eglStreamProducer->connectEGLProducer()) {
        g_print("Connect EGL stream cuda producer failed.\n");
        return -1;
    }

    // Firstly, call cuEGLStreamProducerPresentFrame to push 4 frame buffers to the EGL stream FIFO.
    eglStreamProducer->presentFrameBuffers(4);

    // start the cuda producer
    std::thread t = std::thread(producerThreadFunc);

    t.join();
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
    delete eglStreamProducer;
    return 0;
}