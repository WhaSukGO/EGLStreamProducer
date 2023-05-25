# All common header files
CPPFLAGS += \
	-I/usr/local/cuda-10.2/include \
	-I/usr/include/aarch64-linux-gnu \
	`pkg-config --cflags gstreamer-1.0`

# All common dependent libraries
LDFLAGS += \
	-lGLESv2 -lEGL -lcuda -lcudart \
	-L/usr/local/cuda/lib64 \
	`pkg-config --libs gstreamer-1.0`

SRCS := \
	main.cpp \
	eglstreamproducer.cpp \
	EGLAPIAccessors.cpp

main:
	g++ $(SRCS) $(CPPFLAGS) $(LDFLAGS)
	