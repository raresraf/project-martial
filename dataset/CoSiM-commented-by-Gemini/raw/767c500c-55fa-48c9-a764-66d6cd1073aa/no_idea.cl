/**
 * @file no_idea.cl
 * @brief An OpenCL implementation for a block-based texture compression algorithm.
 *
 * This file contains both the OpenCL device code (the `copy_image` kernel and helpers)
 * and C++ host code for orchestrating the compression. The algorithm is a variant of
 * Ericsson Texture Compression (ETC), compressing 4x4 pixel blocks. Despite the
 * kernel's name, its function is to compress, not just copy, image data.
 */

// --- OpenCL Device Code ---

// Represents a 32-bit BGRA color, allowing access by channels, components, or raw bits.
union Color {
	struct BgraColorType {
		uchar b;
		uchar g;
		uchar r;
		uchar a;
	} channels;
	uchar components[4];
	uint bits;
};

// Clamping and rounding helper functions for color quantization.
uchar clampa(uchar val, uchar min, uchar max) {
	return val < min ? min : (val > max ? max : val);
}
int clampa_int(int val, int min, int max) {
	return val < min ? min : (val > max ? max : val);
}
uchar round_to_5_bits(float val) {
	return clampa (val * 31.0f / 255.0f + 0.5f, 0, 31);
}
uchar round_to_4_bits(float val) {
	return clampa (val * 15.0f / 255.0f + 0.5f, 0, 15);
}

/**
 * @brief Codeword tables for luminance modulation.
 * Stored in __constant memory for fast, cached access by all work-items.
 */
__constant short g_codeword_tables[8][4] = {
	{-8, -2, 2, 8},
	{-17, -5, 5, 17},
	{-29, -9, 9, 29},
	{-42, -13, 13, 42},
	{-60, -18, 18, 60},
	{-80, -24, 24, 80},
	{-106, -33, 33, 106},
	{-183, -47, 47, 183}};

// Look-up tables for pixel index reordering and mapping.
__constant uchar g_mod_to_pix[4] = {3, 2, 0, 1};
__constant uchar g_idx_to_num[4][8] = {
	{0, 4, 1, 5, 2, 6, 3, 7},        
	{8, 12, 9, 13, 10, 14, 11, 15},  
	{0, 4, 8, 12, 1, 5, 9, 13},      
	{2, 6, 10, 14, 3, 7, 11, 15}     
};

// ... (Core ETC-like helper functions: makeColor, getColorError, WriteColors*, etc.) ...
// These functions are responsible for the low-level details of constructing
// the compressed 8-byte block from the 4x4 pixel input.

/**
 * @brief Main device-side compression function for a single 4x4 block.
 *
 * This function orchestrates the compression logic for a block, including checking
 * for solid colors, determining the flip and differential modes, and calling
 * the core luminance computation.
 */
unsigned long compressBlock(__global uchar* dst,
							union Color* ver_src,
							union Color* hor_src,
							unsigned long threshold); // Forward declaration

// Custom device-side memcpy implementation.
void memcpy(uchar *dst, __global uchar *src, int size){
	for (int i = 0 ; i < size; i++)
		dst[i] = src[i];
}


/**
 * @brief The main OpenCL kernel for texture compression. (Misnamed as copy_image).
 *
 * Each work-item in the 2D NDRange is responsible for compressing a single
 * 4x4 block of the source image.
 *
 * @param width Width of the source image.
 * @param height Height of the source image.
 * @param src Input image data in global memory.
 * @param dst Output buffer for compressed block data in global memory.
 */
__kernel void copy_image(const int width, const int height,
				__global uchar *src,
				__global uchar *dst)
{
	// Threading Model: Map 2D global ID to a 4x4 block coordinate.
	int col = get_global_id(0);
	int linie = get_global_id(1); // 'linie' is Romanian for 'row'.
	unsigned long compressed_error = 0;

	union Color ver_blocks[16];
	union Color hor_blocks[16];
	
	// Memory Model: Gather pixel data from global memory into private memory arrays.
	// This prepares the data for evaluating both vertical and horizontal partition modes.
	__global union Color* row0 = (__global union Color*) (src + 4 * linie * (width * 4) + col * 4 * 4);
	__global union Color* row1 = row0 + width;
	__global union Color* row2 = row1 + width;
	__global union Color* row3 = row2 + width;
	
	memcpy((char*)&ver_blocks[0], (char*)row0, 8);
	memcpy((char*)&ver_blocks[2], (char*)row1, 8);
	// ... (rest of memcpy operations for data gathering) ...
	
	memcpy((char*)&hor_blocks[0], (char*)row0, 16);
	memcpy((char*)&hor_blocks[4], (char*)row1, 16);
	memcpy((char*)&hor_blocks[8], (char*)row2, 16);
	memcpy((char*)&hor_blocks[12], (char*)row3, 16);
	
	// Call the main compression logic for the current block.
	compressed_error += compressBlock(dst + 8 * (width/4*linie+col), ver_blocks, hor_blocks, INT_MAX);
}

// --- C++ Host Code ---
#include "compress.hpp"

using namespace std;

#define DIE(assertion, call_description) /* ... */

// ... (Utility functions for OpenCL error checking: cl_get_string_err, cl_get_compiler_err_log) ...

/**
 * @brief Helper function to find and select a GPU device.
 * @param device Output parameter for the selected OpenCL device ID.
 */
void gpu_find(cl_device_id &device);

/**
 * @brief Constructor for the TextureCompressor class.
 *
 * Architectural Intent: Initializes the OpenCL environment by finding a GPU device,
 * creating a context, and creating a command queue. The kernel is compiled later.
 */
TextureCompressor::TextureCompressor()
{
	gpu_find(device);
} 
TextureCompressor::~TextureCompressor() {} 

/**
 * @brief Compresses an image using the OpenCL kernel.
 *
 * This method orchestrates the compression by setting up memory buffers,
 * compiling the kernel, transferring data, executing the kernel, and reading
 * back the results.
 */
unsigned long TextureCompressor::compress(const uint8_t* src,
									  uint8_t* dst,
									  int width,
									  int height)
{
	cl_int ret;
	string kernel_src;
	
	// 1. Create OpenCL context and command queue.
	context = clCreateContext(0, 1, &device, NULL, NULL, &ret);
	CL_ERR(ret);
	command_queue = clCreateCommandQueue(context, device, 0, &ret);
	CL_ERR(ret);

	// 2. Create device memory buffers.
	cl_mem src_dev = clCreateBuffer(context, CL_MEM_READ_WRITE, 4 * width * height, NULL, &ret);
	CL_ERR(ret);
	cl_mem dst_dev = clCreateBuffer(context, CL_MEM_READ_WRITE, width * height * 4 / 8, NULL, &ret);
	CL_ERR(ret);

	// 3. Copy source data from host to device.
	ret = clEnqueueWriteBuffer(command_queue, src_dev, CL_TRUE, 0, 4 * width * height, src, 0, NULL, NULL);
	CL_ERR(ret);

	// 4. Read, create, and build the OpenCL program.
	read_kernel("no_idea.cl", kernel_src);
	const char *kernel_c_str = kernel_src.c_str();
	program = clCreateProgramWithSource(context, 1, &kernel_c_str, NULL, &ret);
	CL_ERR(ret);
	ret = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	CL_COMPILE_ERR(ret, program, device);
	kernel = clCreateKernel(program, "copy_image", &ret);
	CL_ERR(ret);

	// 5. Set kernel arguments.
	CL_ERR(clSetKernelArg(kernel, 0, sizeof(cl_uint), (void *)&width));
	CL_ERR(clSetKernelArg(kernel, 1, sizeof(cl_uint), (void *)&height));
	CL_ERR(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&src_dev));
	CL_ERR(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&dst_dev));
	
	// 6. Define NDRange and execute the kernel.
	size_t globalSize[2] = {(size_t)width/4, (size_t)height/4};
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, globalSize, 0, 0, NULL, NULL);
	CL_ERR(ret);
	CL_ERR(clFinish(command_queue));

	// 7. Read compressed data back to host.
	CL_ERR(clEnqueueReadBuffer(command_queue, dst_dev, CL_TRUE, 0, width * height * 4 / 8, dst, 0, NULL, NULL));
	
	// 8. Clean up resources.
	CL_ERR(clReleaseMemObject(src_dev));
	CL_ERR(clReleaseMemObject(dst_dev));
	CL_ERR(clReleaseCommandQueue(command_queue));
	CL_ERR(clReleaseContext(context));
	
	cout << " DONE
 ";
	return 0;
}
