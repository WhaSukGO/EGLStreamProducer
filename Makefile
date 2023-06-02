# All common header files
CPPFLAGS += \
	-I/usr/local/cuda-10.2/include \
	-I/usr/include/aarch64-linux-gnu \
	`pkg-config --cflags gstreamer-1.0 opencv4`

# All common dependent libraries
LDFLAGS += \
	-lGLESv2 -lEGL -lcuda -lcudart \
	-L/usr/local/cuda/lib64 \
	-lnppc -lnppicc -lnppim -lnppig \
	`pkg-config --libs gstreamer-1.0 opencv4`

SRCS := \
	main.cpp \
	EGLStreamConsumer.cpp \
	EGLAPIAccessors.cpp

main:
	g++ $(SRCS) $(CPPFLAGS) $(LDFLAGS)
	