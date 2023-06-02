# Video to EGLStream
 - A EGLStream producer, a modified version of the code [here](https://forums.developer.nvidia.com/t/frames-returned-from-nveglstreamsrc-via-egl-stream-out-of-order/53074)
 - Written in C++
 - Supports `.mkv` file (`.mp4` is upcoming)

## Notes
 - `nvvideosink` only supports **ABGR**
 - `nvvidconv` cannot control the output format using `bl-output` (block-linear)
