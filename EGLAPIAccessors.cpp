#include <stdio.h>

#include "EGLAPIAccessors.hpp"


EGLDisplay EGLDisplayAccessor::getInstance()
{
    static EGLDisplayAccessor instance;
    return instance.eglDisplay;
}

EGLDisplayAccessor::EGLDisplayAccessor()
{
    // Obtain the EGL display
    if ((eglDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY)) == EGL_NO_DISPLAY) {
        printf("EGL failed to obtain display.\n");
    }

    // Initialize EGL
    EGLint major, minor;
    if (!eglInitialize(eglDisplay, &major, &minor)) {
        printf("EGL failed to initialize.\n");
        eglTerminate(eglDisplay);
        eglDisplay = EGL_NO_DISPLAY;
    } else {
        printf("EGL API: %d.%d\n", major, minor);
    }
}

EGLDisplayAccessor::~EGLDisplayAccessor()
{
    if (eglDisplay != EGL_NO_DISPLAY) {
        eglTerminate(eglDisplay);
        eglDisplay = EGL_NO_DISPLAY;

        printf("Terminate EGL display.\n");
        fflush(stdout);
    }
}


static bool initialized = false;

#define EXTLST_IMPL_MY(tx, x) tx x = nullptr;
EXTENSION_LIST_MY(EXTLST_IMPL_MY)

typedef void (* extlst_fnptr_t)(void);
#define EXTLST_ENTRY_MY(tx, x) { ( extlst_fnptr_t *)&x, #x },

static struct {
    extlst_fnptr_t * fnptr;
    char const * name;
} extensionList[] = { EXTENSION_LIST_MY(EXTLST_ENTRY_MY) };

bool setupEGLExtensions()
{
    if (!initialized) {
        for (size_t i = 0; i < sizeof(extensionList) / sizeof(extensionList[0]); i++) {
            *extensionList[i].fnptr = eglGetProcAddress(extensionList[i].name);
            if (!*extensionList[i].fnptr) {
                printf("Couldn't get address of %s()\n", extensionList[i].name);
                return false;
            }
        }

        initialized = true;
    }

    return true;
}