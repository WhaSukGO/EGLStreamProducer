#include <stdio.h>

#include <gst/gst.h>
#include <cuda_runtime.h>
#include "opencv2/highgui.hpp"

#include "EGLStreamConsumer.hpp"


static void onDemuxPadAdded(GstElement *src, GstPad *pad, GstElement *dst)
{
    GstPad *sink_pad = gst_element_get_static_pad(dst, "sink");
    GstPadLinkReturn ret;
    GstCaps *new_pad_caps = NULL;
    GstStructure *new_pad_struct = NULL;
    const gchar *new_pad_type = NULL;

    /* If our converter is already linked, we have nothing to do here */
    if (gst_pad_is_linked(sink_pad))
    {
        g_print("We are already linked. Ignoring.\n");
        goto exit;
    }

    /* Check the new pad's type */
    new_pad_caps = gst_pad_get_current_caps(pad);
    new_pad_struct = gst_caps_get_structure(new_pad_caps, 0);
    new_pad_type = gst_structure_get_name(new_pad_struct);
    if (!g_str_has_prefix(new_pad_type, "video/x-h265"))
    {
        g_print("It has type '%s' which is not raw audio. Ignoring.\n", new_pad_type);
        goto exit;
    }

    /* Attempt the link */
    ret = gst_pad_link(pad, sink_pad);
    if (GST_PAD_LINK_FAILED(ret))
    {
        g_print("Type is '%s' but link failed.\n", new_pad_type);
    }
    else
    {
        g_print("Link succeeded (type '%s').\n", new_pad_type);
    }

exit:
    /* Unreference the new pad's caps, if we got them */
    if (new_pad_caps != NULL)
        gst_caps_unref(new_pad_caps);

    /* Unreference the sink pad */
    gst_object_unref(sink_pad);
}

int runEGLConsumer()
{
    int width = 1640;
    int height = 1232;

    gst_init(nullptr, nullptr);

    EGLStreamConsumer *eglStreamConsumer = new EGLStreamConsumer(4, 0, width, height);

    GstElement *pipeline = gst_pipeline_new("play");
    GstElement *source = gst_element_factory_make("filesrc", nullptr);
    GstElement *matroskademux = gst_element_factory_make("matroskademux", nullptr);
    GstElement *queue = gst_element_factory_make("queue", nullptr);
    GstElement *h265parse = gst_element_factory_make("h265parse", nullptr);
    GstElement *nvv4l2decoder = gst_element_factory_make("nvv4l2decoder", nullptr);
    GstElement *nvvidconv = gst_element_factory_make("nvvidconv", nullptr);
    GstElement *nvvideosink = gst_element_factory_make("nvvideosink", nullptr);

    if (!pipeline || !source || !matroskademux || !queue || !h265parse || !nvv4l2decoder || !nvvidconv || !nvvideosink)
    {
        g_print("Failed to create elements\n");
        return -1;
    }

    g_object_set(source, "location", "/home/nano1/Development/2023/proj/1.mkv", nullptr);
    g_object_set(nvv4l2decoder, "output-buffers", 4, nullptr);
    g_object_set(nvvideosink, "display", eglStreamConsumer->getEGLDisplay(), nullptr);
    g_object_set(nvvideosink, "stream", eglStreamConsumer->getEGLStream(), nullptr);
    g_object_set(nvvideosink, "fifo", true, nullptr);
    g_object_set(nvvideosink, "fifo-size", 4, nullptr);

    g_signal_connect(matroskademux, "pad-added", G_CALLBACK(&onDemuxPadAdded), queue);

    gst_bin_add_many(GST_BIN(pipeline), source, matroskademux, queue, h265parse, nvv4l2decoder, nvvidconv, nvvideosink, nullptr);

    if (!gst_element_link(source, matroskademux))
    {
        g_print("source -> demux failed.\n");
        return -1;
    }

    if (!gst_element_link_many(queue, h265parse, nvv4l2decoder, nvvidconv, nvvideosink, nullptr))
    {
        g_print("queue -> sink failed.\n");
        return -1;
    }

    GstState state, pending;
    GstStateChangeReturn ret;

    ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE)
    {
        g_print("Change pipeline state to %s failed.\n", gst_element_state_get_name(GST_STATE_PLAYING));
        return -1;
    }

    for (;;)
    {
        ret = gst_element_get_state(pipeline, &state, &pending, 0);
        if (ret == GST_STATE_CHANGE_SUCCESS && state == GST_STATE_PLAYING)
        {
            printf("Ready to retrieve frame\n");
            break;
        }
    }

    cv::Mat cpuMat(cv::Size(width, height), CV_8UC3);

    for (int idx = 0;; idx++)
    {
        printf("%dth frame retrieved\n", idx);

        cv::cuda::GpuMat gpuMat = eglStreamConsumer->acquireFrame();

        if (gpuMat.empty())
            break;

        gpuMat.download(cpuMat);

        cv::imshow("cpuMat", cpuMat);
        cv::waitKey(1);

        cudaFree(gpuMat.data);
        idx++;
    }

    return 0;
}

int main(int argc, char *argv[])
{
    runEGLConsumer();
}