# Video to EGLStream
 - This project is for **Tegra platform** (ex. Jetson Nano), manipulating Nvidia's GStreamer plugins for hardware acceleartion
 - A EGLStream producer, a modified version of the code [here](https://forums.developer.nvidia.com/t/frames-returned-from-nveglstreamsrc-via-egl-stream-out-of-order/53074)
 - Written in C++
 - Supports `.mkv` file (`.mp4` is upcoming)

## Notes
 - `nvvideosink` only supports **ABGR** (check it out on [nvvideosink](https://developer.download.nvidia.com/embedded/L4T/r32_Release_v1.0/Docs/Accelerated_GStreamer_User_Guide.pdf) section)
 - `nvvidconv` cannot control the output format using `bl-output` (block-linear)
