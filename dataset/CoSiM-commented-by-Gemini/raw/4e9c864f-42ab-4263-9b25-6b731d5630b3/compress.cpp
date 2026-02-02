#include "helper.hpp"
#include "compress.hpp"

using namespace std;

TextureCompressor::TextureCompressor() {
	
	cl_platform_id platform;
	cl_uint platform_num = 0;
	cl_uint device_num = 0;

	size_t attr_size = 0;
	cl_char* attr_data = NULL;

	
	CL_ERR(clGetPlatformIDs(0, NULL, &platform_num));
	platform_ids = new cl_platform_id[platform_num];
	DIE(platform_ids == NULL, "alloc platform_ids");

	
	CL_ERR(clGetPlatformIDs(platform_num, platform_ids, NULL));

	
	for (uint platf = 0; platf < platform_num; platf++) {
		platform = platform_ids[platf];
		DIE(platform == 0, "platform selection");

		CL_ERR(clGetDeviceIDs(platform,
			CL_DEVICE_TYPE_GPU, 0, NULL, &device_num));
		device_ids = new cl_device_id[device_num];
		DIE(device_ids == NULL, "alloc devices");

		if (device_num != 0)
			CL_ERR(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU,
				device_num, device_ids, NULL));

		
		if (device_num != 0) {


			device = device_ids[0];
			break;
		}
	}

} 	
TextureCompressor::~TextureCompressor() { }	
	
unsigned long TextureCompressor::compress(const uint8_t* src,
											  uint8_t* dst,
											  int width,
											  int height) {

	uint8_t* copy = dst;
	unsigned long compressed_error = 0;
	string kernel_src;

	cl_int ret;
	int srcSize;
	int dstSize;

	size_t sizeHeight = height / 4;
	size_t sizeWidth = width / 4;	

	
	srcSize = width * height * 4;
	dstSize = width * height * 4 / 8;

	


	context = clCreateContext(0, 1, &device, NULL, NULL, &ret);
	CL_ERR(ret);

	command_queue = clCreateCommandQueue(context, device, 0, &ret);
	CL_ERR(ret);

	
	cl_mem srcBuf = clCreateBuffer(context, CL_MEM_READ_ONLY,
		sizeof(cl_uchar) * srcSize, NULL, &ret);
	CL_ERR(ret);

	cl_mem dstBuf = clCreateBuffer(context, CL_MEM_READ_WRITE,
		sizeof(cl_uchar) * dstSize, NULL, &ret);
	CL_ERR(ret);

	
	CL_ERR(clEnqueueWriteBuffer(command_queue, srcBuf, CL_TRUE, 0, 
		sizeof(uint8_t) * srcSize, src, 0, NULL, NULL));

	CL_ERR(clEnqueueWriteBuffer(command_queue, dstBuf, CL_TRUE, 0, 
		sizeof(uint8_t) * dstSize, dst, 0, NULL, NULL));

	
	read_kernel("compressETC1.cl", kernel_src);
	const char* kernel_c_str;
	kernel_c_str = kernel_src.c_str();

	
	program = clCreateProgramWithSource(context, 1,
		  (const char**) &kernel_c_str, NULL, &ret);
	CL_ERR(ret);

	
	ret = clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", NULL, NULL);
	CL_COMPILE_ERR(ret, program, device);

	
	kernel = clCreateKernel(program, "compressFunc", &ret);
	CL_ERR(ret);

	
	CL_ERR(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&srcBuf));
	CL_ERR(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&dstBuf));
	CL_ERR(clSetKernelArg(kernel, 2, sizeof(int), &width));
	CL_ERR(clSetKernelArg(kernel, 3, sizeof(int), &height));

	
	cl_event event;
	size_t globalSize[2] = {sizeHeight, sizeWidth};
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL,
		  globalSize, 0, 0, NULL, &event);
	CL_ERR(ret);
	CL_ERR(clWaitForEvents(1, &event));

	
	CL_ERR(clEnqueueReadBuffer(command_queue, dstBuf, CL_TRUE, 0,
		sizeof(uint8_t) * dstSize, dst, 0, NULL, NULL));

	
	CL_ERR(clFinish(command_queue));

	
	CL_ERR(clReleaseProgram(program));
	CL_ERR(clReleaseKernel(kernel));
	CL_ERR(clReleaseMemObject(dstBuf));
	CL_ERR(clReleaseMemObject(srcBuf));
	CL_ERR(clReleaseCommandQueue(command_queue));
	CL_ERR(clReleaseContext(context));
	
	return compressed_error;
}
