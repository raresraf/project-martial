/**
 * @file compress.hpp
 * @brief Defines a class for performing texture compression using OpenCL.
 *
 * This file contains the definition of the TextureCompressor class, which encapsulates
 * the necessary OpenCL objects and methods to compress raw image data into a
 * compressed texture format.
 */
#ifndef __TEXTURE_COMPRESSOR__
#define __TEXTURE_COMPRESSOR__

#if __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <stdint.h>

/**
 * @class TextureCompressor
 * @brief Manages OpenCL resources for texture compression.
 *
 * This class sets up the OpenCL environment, including platform, device, context,
 * command queue, and the compression kernel. It provides a simple interface
 to
 * compress a texture.
 */
class TextureCompressor {
public:
    /**
     * @brief Constructs a new TextureCompressor object.
     *
     * Initializes the OpenCL environment, discovers platforms and devices,
     * creates a context and command queue, and builds the compression program
     * and kernel.
     */
    TextureCompressor();

    /**
     * @brief Destroys the TextureCompressor object.
     *
     * Releases all allocated OpenCL resources, including the kernel, program,
     * command queue, context, and device identifiers.
     */
    ~TextureCompressor();

    /**
     * @brief Compresses a raw image.
     * @param src Pointer to the source raw image data.
     * @param dst Pointer to the destination buffer for the compressed data.
     * @param width The width of the source image in pixels.
     * @param height The height of the source image in pixels.
     * @return The size of the compressed data in bytes.
     */
    unsigned long compress(const uint8_t* src, uint8_t* dst, int width, int height);

private:
    cl_platform_id* platform_ids;  ///< Array of available OpenCL platform IDs.
    cl_device_id* device_ids;      ///< Array of available OpenCL device IDs.
    cl_device_id device;           ///< The selected OpenCL device for execution.
    cl_context context;            ///< The OpenCL context for managing resources.
    cl_command_queue command_queue;///< The command queue for submitting kernels.
    cl_program program;            ///< The OpenCL program containing the compression kernel.
    cl_kernel kernel;              ///< The OpenCL kernel that performs the compression.
};

#endif
