/**
 * @file texture_compress_skl.cpp
 * @brief Host-side implementation for OpenCL-accelerated texture compression.
 * 
 * Architectural Intent: Manages OpenCL lifecycle including device discovery, 
 * resource allocation, and kernel execution for bulk image processing.
 */

#include "compress.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <CL/cl.h>

#include "helper.cpp"

using namespace std;

static int platform_select;

/**
 * @brief Scans the system for OpenCL platforms and selects the target GPU device.
 * @param device Output pointer for the selected device ID.
 * @param platform_select Index of the platform to prefer.
 * @param device_select Index of the device on the platform to select.
 * @param device_ids Buffer for available device IDs.
 * @param platform_ids Buffer for available platform IDs.
 */
void gpu_find(cl_device_id *device, 
	uint platform_select, 
	uint device_select, cl_device_id *device_ids, cl_platform_id *platform_ids)
{
	cl_platform_id platform;
	cl_uint platform_num = 0;
	cl_uint device_num = 0;
	size_t attr_size = 0;
	cl_char* attr_data = NULL;

	/* get num of available OpenCL platforms */
	CL_ERR( clGetPlatformIDs(0, NULL, &platform_num));
	platform_ids = new cl_platform_id[platform_num];
	DIE(platform_ids == NULL, "alloc platform_list");

	/* get all available OpenCL platforms */
	CL_ERR( clGetPlatformIDs(platform_num, platform_ids, NULL));
	cout << "Platforms found: " << platform_num << endl;

	/**
	 * Block Logic: Platform enumeration and discovery.
	 * Invariant: Iterates through all detected OpenCL stacks on the host.
	 */
	for(uint platf=0; platf<platform_num; platf++)
	{
		/* get attribute CL_PLATFORM_VENDOR */
		CL_ERR( clGetPlatformInfo(platform_ids[platf],
				CL_PLATFORM_VENDOR, 0, NULL, &attr_size));
		attr_data = new cl_char[attr_size];
		DIE(attr_data == NULL, "alloc attr_data");

		/* get data CL_PLATFORM_VENDOR */
		CL_ERR( clGetPlatformInfo(platform_ids[platf],
				CL_PLATFORM_VENDOR, attr_size, attr_data, NULL));
		cout << "Platform " << platf << " " << attr_data << " ";
		delete[] attr_data;

		/* get attribute size CL_PLATFORM_VERSION */
		CL_ERR( clGetPlatformInfo(platform_ids[platf],
				CL_PLATFORM_VERSION, 0, NULL, &attr_size));
		attr_data = new cl_char[attr_size];
		DIE(attr_data == NULL, "alloc attr_data");

		/* get data size CL_PLATFORM_VERSION */
		CL_ERR( clGetPlatformInfo(platform_ids[platf],
				CL_PLATFORM_VERSION, attr_size, attr_data, NULL));
		cout << attr_data << endl;
		delete[] attr_data;

		/* no valid platform found */
		platform = platform_ids[platf];
		DIE(platform == 0, "platform selection");

		/**
		 * Block Logic: Device filtering within platform.
		 * Pre-condition: Valid platform ID selected.
		 * Invariant: Populates 'device_ids' with available GPU compute units.
		 */
		if(clGetDeviceIDs(platform, 
			CL_DEVICE_TYPE_GPU, 0, NULL, &device_num) == CL_DEVICE_NOT_FOUND) {
			device_num = 0;
			continue;
		}

		device_ids = new cl_device_id[device_num];
		DIE(device_ids == NULL, "alloc devices");

		/* get all available OpenCL devices type ALL on the selected platform */
		CL_ERR( clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU,
			device_num, device_ids, NULL));
		cout << "\tDevices found " << device_num  << endl;
		platform_select = platf; 

		/**
		 * Block Logic: Device selection and metadata logging.
		 * Invariant: Selects the specific compute device requested via indices.
		 */
		for(uint dev=0; dev<device_num; dev++)
		{
			/* get attribute size */
			CL_ERR( clGetDeviceInfo(device_ids[dev], CL_DEVICE_NAME,
				0, NULL, &attr_size));
			attr_data = new cl_char[attr_size];
			DIE(attr_data == NULL, "alloc attr_data");

			/* get attribute CL_DEVICE_NAME */
			CL_ERR( clGetDeviceInfo(device_ids[dev], CL_DEVICE_NAME,
				attr_size, attr_data, NULL));
			cout << "\tDevice " << dev << " " << attr_data << " ";
			delete[] attr_data;

			/* get attribute size */
			CL_ERR( clGetDeviceInfo(device_ids[dev], CL_DEVICE_VERSION,
				0, NULL, &attr_size));
			attr_data = new cl_char[attr_size];
			DIE(attr_data == NULL, "alloc attr_data");

			/* get attribute CL_DEVICE_VERSION */
			CL_ERR( clGetDeviceInfo(device_ids[dev], CL_DEVICE_VERSION,
				attr_size, attr_data, NULL));
			cout << attr_data; 
			delete[] attr_data;

			/* select device based on cli arguments */
			if((platf == platform_select) && (dev == device_select)){
				*device = device_ids[dev];
				cout << " <--- SELECTED ";
			}

			cout << endl;
		}
	}
}

/**
 * @brief Initializes the TextureCompressor and selects the compute device.
 */
TextureCompressor::TextureCompressor() {
	gpu_find(&device, platform_select, 1, device_ids, platform_ids);
}

TextureCompressor::~TextureCompressor() { 
}

/**
 * @brief Executes the texture compression pipeline on the GPU.
 * @param src Host buffer containing raw image data.
 * @param dst Host buffer to receive compressed data.
 * @param width Image width.
 * @param height Image height.
 * @return 0 on success.
 */
unsigned long TextureCompressor::compress(const uint8_t* src,
									  uint8_t* dst,
									  int width,
									  int height)
{
    cl_int ret;
    string kernel_src;
    cl_mem bufSource, bufDest;

	// Functional Utility: Establishes the OpenCL context for the selected hardware device.
    context = clCreateContext(0, 1, &device, NULL, NULL, &ret);
    CL_ERR(ret);

	// Functional Utility: Creates a command queue to orchestrate GPU operations.
    command_queue = clCreateCommandQueue(context, device, 0, &ret);
    CL_ERR(ret);

	// Resource Management: Allocates read-only global memory for the source image.
    bufSource = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_uchar) * width * height *  4, NULL, &ret);

	// Resource Management: Allocates read-write global memory for the compressed output.
    bufDest = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_uchar) * width  * height / 2, NULL, &ret);
    CL_ERR(ret);

	// Compilation: Loads kernel source from filesystem and triggers runtime compilation.
    read_kernel("kernel_tiberiu4.cl", kernel_src);
    const char *kernel_c_str = kernel_src.c_str();
    program = clCreateProgramWithSource(context, 1,(const char **) &kernel_c_str, NULL, &ret);
    CL_ERR(ret);

	// Optimization: Builds the program binary for the target device architecture.
    ret = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    CL_COMPILE_ERR(ret, program, device);

	// Kernel Instantiation: Creates the entry point for image compression logic.
    kernel = clCreateKernel(program, "imgCompress", &ret);
    CL_ERR(ret);

	// Marshalling: Binds host-allocated memory and image dimensions as kernel arguments.
    CL_ERR(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &bufSource));
    CL_ERR(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &bufDest));
    CL_ERR(clSetKernelArg(kernel, 2, sizeof(cl_int), (void *) &width));
    CL_ERR(clSetKernelArg(kernel, 3, sizeof(cl_int), (void *) &height));

	// Data Migration: Asynchronously transfers image data from host RAM to GPU global memory.
    clEnqueueWriteBuffer(command_queue, bufSource, CL_TRUE, 0, sizeof(cl_uchar) * width * height *4, src, 0, NULL, NULL);

	// Dispatch: Launches the parallel compute grid based on 4x4 block granularity.
    size_t globalSize[2] = {(size_t)height/4,(size_t) width/4};
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, globalSize, NULL, 0, NULL, NULL);

	// Synchronization: Blocks host execution until the GPU task queue is exhausted.
    CL_ERR(clFinish(command_queue));

	// Data Recovery: Retrieves compressed results from GPU memory back to host RAM.
	CL_ERR(clEnqueueReadBuffer(command_queue, bufDest, CL_TRUE, 0, sizeof(cl_uchar) * width * height / 2, dst, 0, NULL, NULL));

	// Lifecycle Cleanup: Releases OpenCL handle objects to prevent resource leaks.
	CL_ERR(clReleaseProgram(program));
	CL_ERR(clReleaseKernel(kernel));
	CL_ERR(clReleaseMemObject(bufSource));
	CL_ERR(clReleaseMemObject(bufDest));
	CL_ERR(clReleaseContext(context));
	CL_ERR(clReleaseCommandQueue(command_queue));
	return 0;
}
