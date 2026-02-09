/**
 * @file texture_compress_mohamad.cpp
 * @brief Implements an OpenCL-based texture compressor with a dedicated device finder.
 *
 * This file contains the implementation of the TextureCompressor class. The logic
 * for discovering and selecting an OpenCL device is refactored into a helper
 * function, `gpu_find`.
 */
#include "compress.hpp"

#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

using namespace std;

/**
 * @brief Finds and selects a specific OpenCL GPU device.
 * @param[out] device Reference to a cl_device_id that will hold the selected device.
 * @param[in] platform_select The index of the platform to select.
 * @param[in] device_select The index of the device on that platform to select.
 *
 * This function iterates through all available OpenCL platforms and their GPU
 * devices, selecting the device that matches the specified platform and device indices.
 * @note This function has memory leaks. The `attr_data`, `platform_list`, and
 *       `device_list` are not always correctly deallocated, especially if
 *       multiple platforms with GPUs exist.
 */
void gpu_find(cl_device_id &device, 
		uint platform_select, 
		uint device_select)
{
	cl_platform_id platform;
	cl_uint platform_num = 0;
	cl_platform_id* platform_list = NULL;

	cl_uint device_num = 0;
	cl_device_id* device_list = NULL;

	size_t attr_size = 0;
	cl_char* attr_data = NULL;

	/* get num of available OpenCL platforms */
	( clGetPlatformIDs(0, NULL, &platform_num));
	platform_list = new cl_platform_id[platform_num];

	/* get all available OpenCL platforms */
	( clGetPlatformIDs(platform_num, platform_list, NULL));

	/* list all platforms and VENDOR/VERSION properties */
	for(uint platf=0; platf<platform_num; platf++)
	{
		/* get and discard attribute CL_PLATFORM_VENDOR */
		( clGetPlatformInfo(platform_list[platf],
				CL_PLATFORM_VENDOR, 0, NULL, &attr_size));
		attr_data = new cl_char[attr_size];

		( clGetPlatformInfo(platform_list[platf],
				CL_PLATFORM_VENDOR, attr_size, attr_data, NULL));
		delete[] attr_data;

		/* get and discard attribute CL_PLATFORM_VERSION */
		( clGetPlatformInfo(platform_list[platf],
				CL_PLATFORM_VERSION, 0, NULL, &attr_size));
		attr_data = new cl_char[attr_size];

		( clGetPlatformInfo(platform_list[platf],
				CL_PLATFORM_VERSION, attr_size, attr_data, NULL));
		delete[] attr_data;

		platform = platform_list[platf];

		/* get num of available OpenCL devices type GPU on the selected platform */
		if(clGetDeviceIDs(platform, 
			CL_DEVICE_TYPE_GPU, 0, NULL, &device_num) == CL_DEVICE_NOT_FOUND) {
			device_num = 0;
			continue;
		}

		device_list = new cl_device_id[device_num];

		/* get all available OpenCL devices type GPU on the selected platform */
		( clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU,
			  device_num, device_list, NULL));

		/* list all devices and TYPE/VERSION properties */
		for(uint dev=0; dev<device_num; dev++)
		{
			/* get and discard attribute CL_DEVICE_NAME */
			( clGetDeviceInfo(device_list[dev], CL_DEVICE_NAME,
				0, NULL, &attr_size));
			attr_data = new cl_char[attr_size];

			( clGetDeviceInfo(device_list[dev], CL_DEVICE_NAME,
				attr_size, attr_data, NULL));
			delete[] attr_data;

			/* get and discard attribute CL_DEVICE_VERSION */
			( clGetDeviceInfo(device_list[dev], CL_DEVICE_VERSION,
				0, NULL, &attr_size));
			attr_data = new cl_char[attr_size];

			( clGetDeviceInfo(device_list[dev], CL_DEVICE_VERSION,
				attr_size, attr_data, NULL)); 
			delete[] attr_data;

			/* select device based on the provided indices */
			if((platf == platform_select) && (dev == device_select)){
				device = device_list[dev];
			}

		}
	}

	delete[] platform_list;
	delete[] device_list;
}

/**
 * @class TextureCompressor
 * @brief Manages OpenCL initialization for texture compression.
 */
TextureCompressor::TextureCompressor() {
    /**
     * Functional Utility: Initializes the compressor by selecting the first
     * available GPU on the first available OpenCL platform.
     */
	gpu_find(this->device, 0, 0);
}

/**
 * @brief Destructor for the TextureCompressor class.
 */
TextureCompressor::~TextureCompressor() { }
	
/**
 * @brief Compresses a raw texture image.
 * @param src Pointer to the source image data.
 * @param dst Pointer to the destination buffer for compressed data.
 * @param width The width of the source image.
 * @param height The height of the source image.
 * @return The size of the compressed data in bytes.
 *
 * @note This function is currently a stub and does not perform any compression.
 */
unsigned long TextureCompressor::compress(const uint8_t* src,
									  uint8_t* dst,
									  int width,
									  int height)
{

	return 0;
}