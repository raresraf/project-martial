/**
 * @file texture_compress_mohamad.cpp
 * @brief An OpenCL-based texture compressor implementation.
 * @details This file contains the implementation for discovering an OpenCL GPU device
 *          and a class structure for a texture compressor. The core compression
 *          logic is currently a stub.
 */
#include "compress.hpp"

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

/**
 * @brief Finds and selects an OpenCL GPU device.
 * @details This function iterates through all available OpenCL platforms and their associated GPU devices.
 *          It selects a device based on the platform and device index provided by the user.
 *          The function is not robust against multiple platforms as it leaks device_list for all but the last platform.
 * @param[out] device Reference to a cl_device_id to store the selected device.
 * @param[in] platform_select The index of the platform to select.
 * @param[in] device_select The index of the GPU device on the selected platform.
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

	/* Block Logic: Iterate through all platforms to find available GPU devices. */
	for(uint platf=0; platf<platform_num; platf++)
	{
		/* get attribute CL_PLATFORM_VENDOR */
		( clGetPlatformInfo(platform_list[platf],
				CL_PLATFORM_VENDOR, 0, NULL, &attr_size));
		attr_data = new cl_char[attr_size];

		/* get data CL_PLATFORM_VENDOR */
		( clGetPlatformInfo(platform_list[platf],
				CL_PLATFORM_VENDOR, attr_size, attr_data, NULL));
		delete[] attr_data;

		/* get attribute size CL_PLATFORM_VERSION */
		( clGetPlatformInfo(platform_list[platf],
				CL_PLATFORM_VERSION, 0, NULL, &attr_size));
		attr_data = new cl_char[attr_size];

		/* get data size CL_PLATFORM_VERSION */
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

		// Potential memory leak: device_list is reallocated in each iteration
		// without being deleted, leaking memory for all but the last platform.
		device_list = new cl_device_id[device_num];

		/* get all available OpenCL devices type GPU on the selected platform */
		( clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU,
			  device_num, device_list, NULL));

		/* Block Logic: Iterate through all devices on the current platform. */
		for(uint dev=0; dev<device_num; dev++)
		{
			/* get attribute size */
			( clGetDeviceInfo(device_list[dev], CL_DEVICE_NAME,
				0, NULL, &attr_size));
			attr_data = new cl_char[attr_size];

			/* get attribute CL_DEVICE_NAME */
			( clGetDeviceInfo(device_list[dev], CL_DEVICE_NAME,
				attr_size, attr_data, NULL));
			delete[] attr_data;

			/* get attribute size */
			( clGetDeviceInfo(device_list[dev], CL_DEVICE_VERSION,
				0, NULL, &attr_size));
			attr_data = new cl_char[attr_size];

			/* get attribute CL_DEVICE_VERSION */
			( clGetDeviceInfo(device_list[dev], CL_DEVICE_VERSION,
				attr_size, attr_data, NULL)); 
			delete[] attr_data;

			/* Pre-condition: Select the device if it matches the specified platform and device indices. */
			if((platf == platform_select) && (dev == device_select)){
				device = device_list[dev];
			}

		}
	}

	delete[] platform_list;
	delete[] device_list; // Only frees the list from the last platform iteration.
}

/**
 * @brief Constructs a TextureCompressor object.
 * @details The constructor initializes the compressor by finding and selecting the default
 *          OpenCL GPU (the first device on the first platform).
 */
TextureCompressor::TextureCompressor() { 
	gpu_find(this->device, 0, 0);
} 	// constructor/Users/grigore.lupescu/Desktop/RESEARCH/asc/teme/tema3/2018/Tema3-schelet/src/compress.cpp

/**
 * @brief Destroys the TextureCompressor object.
 */
TextureCompressor::~TextureCompressor() { }	// destructor
	
/**
 * @brief Compresses a source texture image.
 * @warning This function is a stub and does not contain any compression logic.
 * @param src Pointer to the raw source image data.
 * @param dst Pointer to the destination buffer for compressed data.
 * @param width The width of the source image in pixels.
 * @param height The height of the source image in pixels.
 * @return 0, as the function is not implemented.
 */
unsigned long TextureCompressor::compress(const uint8_t* src,
									  uint8_t* dst,
									  int width,
									  int height)
{
	// This is a stub function. No compression is performed.
	return 0;
}