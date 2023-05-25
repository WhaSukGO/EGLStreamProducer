#ifndef EGLAPIACCESSORS_HPP
#define EGLAPIACCESSORS_HPP

#include <EGL/egl.h>
#include <EGL/eglext.h>

#if !defined EGL_KHR_stream || !defined EGL_KHR_stream_fifo || !defined EGL_KHR_stream_consumer_gltexture
# error "EGL_KHR_stream extensions are not supported!"
#endif

class EGLDisplayAccessor
{
public:
    static EGLDisplay getInstance();

private:
    EGLDisplayAccessor();
    ~EGLDisplayAccessor();

    EGLDisplay eglDisplay;
};

#define EXTENSION_LIST_MY(T)                                     \
    T( PFNEGLCREATESTREAMKHRPROC,          eglCreateStreamKHR )  \
    T( PFNEGLDESTROYSTREAMKHRPROC,         eglDestroyStreamKHR ) \
    T( PFNEGLQUERYSTREAMKHRPROC,           eglQueryStreamKHR )   \
    T( PFNEGLSTREAMATTRIBKHRPROC,          eglStreamAttribKHR )


#define EXTLST_EXTERN(tx, x) extern tx x;

EXTENSION_LIST_MY(EXTLST_EXTERN)

bool setupEGLExtensions();


#endif // EGLAPIACCESSORS_HPP