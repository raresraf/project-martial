
>>>> file: texture_compress_skl.cpp
#include "compress.hpp"

#include 
#include 
#include 
#include 
#include 
#include 

using namespace std;

TextureCompressor::TextureCompressor() {
	cl_platform_id 	platform;

	cl_uint 		platform_num = 0;
	cl_uint 		device_num = 0;
	
	size_t 			attr_size = 0;


	cl_char* 		attr_data = NULL;

	/* get num of available OpenCL platforms */
	clGetPlatformIDs(0, NULL, &platform_num);
	platform_ids = new cl_platform_id[platform_num];
	
	/* get all available OpenCL platforms */
	clGetPlatformIDs(platform_num, platform_ids, NULL);

	/* list all platforms and VENDOR/VERSION properties */
	for(uint platf=0; platf<platform_num; platf++)
	{
		/* get attribute CL_PLATFORM_VENDOR */
		clGetPlatformInfo(platform_ids[platf],
			CL_PLATFORM_VENDOR, 0, NULL, &attr_size);
		attr_data = new cl_char[attr_size];
		
		/* get data CL_PLATFORM_VENDOR */
		clGetPlatformInfo(platform_ids[platf],
				CL_PLATFORM_VENDOR, attr_size, attr_data, NULL);
		delete[] attr_data;

		/* get attribute size CL_PLATFORM_VERSION */
		clGetPlatformInfo(platform_ids[platf],
				CL_PLATFORM_VERSION, 0, NULL, &attr_size);
		attr_data = new cl_char[attr_size];
		
		/* get data size CL_PLATFORM_VERSION */
		clGetPlatformInfo(platform_ids[platf],
				CL_PLATFORM_VERSION, attr_size, attr_data, NULL);
		delete[] attr_data;

		/* no valid platform found */
		platform = platform_ids[platf];
		
		/* get num of available OpenCL devices type GPU on the selected platform */
		if(clGetDeviceIDs(platform, 
			CL_DEVICE_TYPE_GPU, 0, NULL, &device_num) == CL_DEVICE_NOT_FOUND) {
			device_num = 0;
			continue;
		}

		device_ids = new cl_device_id[device_num];

		/* get all available OpenCL devices type GPU on the selected platform */
		clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU,
			  device_num, device_ids, NULL);

		/* list all devices and TYPE/VERSION properties */
		for(uint dev=0; dev<device_num; dev++)
		{
			/* get attribute size */
			clGetDeviceInfo(device_ids[dev], CL_DEVICE_NAME,
				0, NULL, &attr_size);
			attr_data = new cl_char[attr_size];

			/* get attribute CL_DEVICE_NAME */
			clGetDeviceInfo(device_ids[dev], CL_DEVICE_NAME,
				attr_size, attr_data, NULL);
			delete[] attr_data;

			/* get attribute size */
			clGetDeviceInfo(device_ids[dev], CL_DEVICE_VERSION,
				0, NULL, &attr_size);
			attr_data = new cl_char[attr_size];

			/* get attribute CL_DEVICE_VERSION */
			clGetDeviceInfo(device_ids[dev], CL_DEVICE_VERSION,
				attr_size, attr_data, NULL);
			delete[] attr_data;

			/* select device based on cli arguments */
			if((platf == 0) && (dev == 0)){
				device = device_ids[dev];
			}
		}
	}
}
TextureCompressor::~TextureCompressor() { }

unsigned long TextureCompressor::compress(const uint8_t* src,
									  uint8_t* dst,
									  int width,
									  int height)
{
	// TODO


	return 0;
}