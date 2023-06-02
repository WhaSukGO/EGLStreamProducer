# Video to EGLStream

 - Written in C++
 - Supports `.mkv` file (`.mp4` is upcoming)

## Notes
 - `nvvideosink` only supports **ABGR**
 - `nvvidconv` cannot control the output format using `bl-output` (block-linear)

## Reference
 - [Nvidia Forum #1](https://forums.developer.nvidia.com/t/frames-returned-from-nveglstreamsrc-via-egl-stream-out-of-order/53074)