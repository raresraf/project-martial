/**
 * @file texture_compress_skl.cpp
 * @brief Implements an OpenCL-based texture compressor, targeting a GPU device.
 *
 * This file contains the implementation of the TextureCompressor class, which is
 * designed to use OpenCL for texture compression. The constructor handles the
 * detection and selection of an appropriate OpenCL-enabled GPU.
 */
#include "compress.hpp"

#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

using namespace std;

/**
 * @class TextureCompressor
 * @brief Manages OpenCL initialization for texture compression.
 *
 * The class constructor probes the system for available OpenCL platforms and
 * devices, selecting a GPU to be used for subsequent compression tasks.
 */
TextureCompressor::TextureCompressor() {
	cl_platform_id 	platform;

	cl_uint 		platform_num = 0;
	cl_uint 		device_num = 0;
	
	size_t 			attr_size = 0;


	cl_char* 		attr_data = NULL;

	/**
	 * Block Logic: Discover available OpenCL platforms.
	 * First call gets the number of platforms, second call retrieves the platform IDs.
	 */
	clGetPlatformIDs(0, NULL, &platform_num);
	platform_ids = new cl_platform_id[platform_num];
	
	clGetPlatformIDs(platform_num, platform_ids, NULL);

	/**
	 * Block Logic: Iterate through all found OpenCL platforms to find a suitable GPU device.
	 * The loop queries vendor and version for each platform, but the information
	 * is not used for selection logic.
	 */
	for(uint platf=0; platf<platform_num; platf++)
	{
		// Get and discard CL_PLATFORM_VENDOR information.
		clGetPlatformInfo(platform_ids[platf],
			CL_PLATFORM_VENDOR, 0, NULL, &attr_size);
		attr_data = new cl_char[attr_size];
		
		clGetPlatformInfo(platform_ids[platf],
				CL_PLATFORM_VENDOR, attr_size, attr_data, NULL);
		delete[] attr_data;

		// Get and discard CL_PLATFORM_VERSION information.
		clGetPlatformInfo(platform_ids[platf],
				CL_PLATFORM_VERSION, 0, NULL, &attr_size);
		attr_data = new cl_char[attr_size];
		
		clGetPlatformInfo(platform_ids[platf],
				CL_PLATFORM_VERSION, attr_size, attr_data, NULL);
		delete[] attr_data;

		/**
		 * Pre-condition: An OpenCL platform is selected to be queried for GPU devices.
		 * The code unconditionally selects the current platform in the iteration.
		 */
		platform = platform_ids[platf];
		
		/**
		 * Block Logic: Check for the presence of any GPU devices on the selected platform.
		 * If no GPU is found, it skips to the next platform.
		 */
		if(clGetDeviceIDs(platform, 
			CL_DEVICE_TYPE_GPU, 0, NULL, &device_num) == CL_DEVICE_NOT_FOUND) {
			device_num = 0;
			continue;
		}

		device_ids = new cl_device_id[device_num];

		// Retrieve the handles for all available GPU devices.
		clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU,
			  device_num, device_ids, NULL);

		/**
		 * Block Logic: Iterate through all found GPU devices on the current platform.
		 * The code queries device name and version but only uses this information
		 * for a hardcoded selection of the very first device found.
		 */
		for(uint dev=0; dev<device_num; dev++)
		{
			// Get and discard CL_DEVICE_NAME information.
			clGetDeviceInfo(device_ids[dev], CL_DEVICE_NAME,
				0, NULL, &attr_size);
			attr_data = new cl_char[attr_size];

			clGetDeviceInfo(device_ids[dev], CL_DEVICE_NAME,
				attr_size, attr_data, NULL);
			delete[] attr_data;

			// Get and discard CL_DEVICE_VERSION information.
			clGetDeviceInfo(device_ids[dev], CL_DEVICE_VERSION,
				0, NULL, &attr_size);
			attr_data = new cl_char[attr_size];

			clGetDeviceInfo(device_ids[dev], CL_DEVICE_VERSION,
				attr_size, attr_data, NULL);
			delete[] attr_data;

			/**
			 * Block Logic: Selects the OpenCL device.
			 * Invariant: The device selected is always the first device (dev == 0) on the
			 * first platform (platf == 0) that has a GPU.
			 */
			if((platf == 0) && (dev == 0)){
				device = device_ids[dev];
			}
		}
	}
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
	// TODO: Compression logic to be implemented.


	return 0;
}