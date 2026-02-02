#ifndef __TEXTURE_COMPRESSOR__
#define __TEXTURE_COMPRESSOR__

#if __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <stdint.h>

class TextureCompressor {
public:
    TextureCompressor();
    ~TextureCompressor();

    unsigned long compress(const uint8_t* src, uint8_t* dst, int width, int height);

private:
    cl_platform_id* platform_ids;
    cl_device_id* device_ids;
    cl_device_id device;
    cl_context context;
    cl_command_queue command_queue;
    cl_program program;
    cl_kernel kernel;
};

#endif
