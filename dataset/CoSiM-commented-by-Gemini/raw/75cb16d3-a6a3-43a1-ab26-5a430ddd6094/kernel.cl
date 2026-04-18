/**
 * @file kernel.cl
 * @brief An OpenCL implementation for a block-based texture compression algorithm.
 *
 * This file contains both the OpenCL device code (kernels and helper functions)
 * for texture compression and the C++ host code for orchestrating the process.
 * The algorithm is a variant of Ericsson Texture Compression (ETC), compressing
 * 4x4 pixel blocks using differential color encoding and luminance modulation.
 */

// --- OpenCL Device Code ---

// Define standard integer types for clarity and portability.
#define INT32_MAX	2147483647
#define UINT32_MAX 	0xffffffff

typedef uchar	uint8_t;
typedef short 	int16_t;
typedef uint 	uint32_t;

// Represents a 32-bit BGRA color, allowing access by channels, components, or raw bits.
typedef union u_Color {
	struct BgraColorType {
		uint8_t b;
		uint8_t g;
		uint8_t r;
		uint8_t a;
	} channels;
	uint8_t components[4];
	uint32_t bits;
} Color;

#define ALIGNAS(X)	__attribute__((aligned(X)))

// Custom device-side memcpy implementation.
void memcpy(void *destination, void *source, size_t num)
{
	char *c_destination = (char *) destination;
	char *c_source = (char *) source;

	for (int i = 0; i < num; i++)
		c_destination[i] = c_source[i];
}

// Custom device-side memset implementation for global memory.
void memset(__global void *ptr, int value, size_t num)
{
	__global char *c_ptr = (__global char *) ptr;
	while (num > 0) {
		*c_ptr = (unsigned char) value;
		c_ptr++;
		num--;
	}
}

// Helper functions for color quantization.
uint8_t round_to_5_bits(float val) {
	return (uint8_t) clamp(val * 31.0f / 255.0f + 0.5f, 0.0f, 31.0f);
}
uint8_t round_to_4_bits(float val) {
	return (uint8_t) clamp(val * 15.0f / 255.0f + 0.5f, 0.0f, 15.0f);
}


/**
 * @brief Codeword tables for luminance modulation.
 * Stored in __constant memory for fast, cached access by all work-items.
 */
ALIGNAS(16) __constant int16_t g_codeword_tables[8][4] = {
	{-8, -2, 2, 8},
	{-17, -5, 5, 17},
	{-29, -9, 9, 29},
	{-42, -13, 13, 42},
	{-60, -18, 18, 60},
	{-80, -24, 24, 80},
	{-106, -33, 33, 106},
	{-183, -47, 47, 183}};


// Look-up tables for pixel index reordering and mapping.
__constant uint8_t g_mod_to_pix[4] = {3, 2, 0, 1};
__constant uint8_t g_idx_to_num[4][8] = {
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
 * @param dst Pointer to the 8-byte destination block in global memory.
 * @param ver_src The 16 source pixels arranged for vertical split evaluation.
 * @param hor_src The 16 source pixels arranged for horizontal split evaluation.
 * @param threshold An error threshold for early termination.
 * @return The calculated compression error for the block.
 */
unsigned long compressBlock(__global uint8_t* dst,
							Color* ver_src,
							Color* hor_src,
							unsigned long threshold); // Forward declaration

/**
 * @brief The main OpenCL kernel for texture compression.
 *
 * Each work-item in the 2D NDRange is responsible for compressing a single
 * 4x4 block of the source image.
 *
 * @param src Input image data in global memory.
 * @param dst Output buffer for compressed block data in global memory.
 * @param width Width of the source image.
 * @param height Height of the source image.
 */
__kernel void compress(__global uchar *src,
					   __global uchar *dst,
					   int width,
					   int height)
{
	Color ver_blocks[16];
	Color hor_blocks[16];

	// Threading Model: Map 2D global ID to a 4x4 block coordinate.
	int y = get_global_id(0);
	int x = get_global_id(1);

	// Calculate memory offsets for the source and destination blocks.
	int offset_src = y * width * 4 * 4 + x * 4 * 4;
	int offset_dst = x * 8 + y * (width / 4) * 8;

	// Memory Model: Gather pixel data from global memory into private memory (ver_blocks, hor_blocks).
	// This prepares the data for evaluating both vertical and horizontal partition modes.
	Color* row0 = (Color*)(src + offset_src);
	Color* row1 = row0 + width;
	Color* row2 = row1 + width;
	Color* row3 = row2 + width;
	
	memcpy(ver_blocks, row0, 8);
	memcpy(ver_blocks + 2, row1, 8);
	// ... (rest of memcpy operations for data gathering) ...
	
	memcpy(hor_blocks, row0, 16);
	memcpy(hor_blocks + 4, row1, 16);
	memcpy(hor_blocks + 8, row2, 16);
	memcpy(hor_blocks + 12, row3, 16);
	
	// Call the main compression logic for the current block.
	compressBlock(dst + offset_dst, ver_blocks, hor_blocks, INT32_MAX);
}


// --- C++ Host Code ---
// The following is C++ code for the host application that uses the OpenCL kernel above.

#include "compress.hpp"

#define DIE(assertion, call_description)  
do { 
	if (assertion) { 
		fprintf(stderr, "(%d): ", __LINE__); 
		perror(call_description); 
		exit(EXIT_FAILURE); 
	} 
} while(0);

using namespace std;

// ... (Utility functions for OpenCL error checking: cl_get_string_err, cl_get_compiler_err_log) ...

/**
 * @brief Helper function to find and select a GPU device.
 * @param device Output parameter for the selected OpenCL device ID.
 */
void gpu_find(cl_device_id &device);

/**
 * @brief Constructor for the TextureCompressor class.
 *
 * Architectural Intent: Initializes the OpenCL environment. It finds a GPU,
 * creates a context and command queue, and pre-compiles the OpenCL kernel.
 */
TextureCompressor::TextureCompressor()
{
	gpu_find(device);
    // ... (Standard OpenCL context, queue, program, and kernel creation) ...
}

TextureCompressor::~TextureCompressor() {
    // ... (Standard OpenCL resource cleanup) ...
}
	
/**
 * @brief Compresses an image using the OpenCL kernel.
 *
 * This method orchestrates the compression by setting up memory buffers,
 * transferring data, executing the kernel, and reading back the results.
 */
unsigned long TextureCompressor::compress(const uint8_t* src,
												uint8_t* dst,
												int width,
												int height) {
	cl_int ret;

	// 1. Create OpenCL context and command queue.
	context = clCreateContext(0, 1, &device, NULL, NULL, &ret);
	command_queue = clCreateCommandQueue(context, device, 0, &ret);

	// 2. Create device memory buffers for source and destination.
	cl_mem bufSrc = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(uint8_t) * width * height * 4, NULL, &ret);
	cl_mem bufDst = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(uint8_t) * width * height * 4, NULL, &ret);

	// 3. Copy source image from host to device buffer.
	ret = clEnqueueWriteBuffer(command_queue, bufSrc, CL_TRUE, 0, sizeof(uint8_t) * width * height * 4, src, 0, NULL, NULL);

	// 4. Read, create, and build the OpenCL program from the "kernel.cl" file.
	string kernel_src;
	read_kernel("kernel.cl", kernel_src);
	const char* kernel_c_str = kernel_src.c_str();
	program = clCreateProgramWithSource(context, 1, &kernel_c_str, NULL, &ret);
	ret = clBuildProgram(program, 1, &device, "", NULL, NULL);
	kernel = clCreateKernel(program, "compress", &ret);
	
	// 5. Set the arguments for the kernel.
	CL_ERR(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &bufSrc));
	CL_ERR(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &bufDst));
	CL_ERR(clSetKernelArg(kernel, 2, sizeof(cl_uint), (void *) &width));
	CL_ERR(clSetKernelArg(kernel, 3, sizeof(cl_uint), (void *) &height));

	// 6. Define the 2D NDRange (work grid) and execute the kernel.
	size_t globalSize[2] = {(size_t) (width / 4), (size_t) (height / 4)};
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, globalSize, 0, 0, NULL, NULL);

	// 7. Read the compressed data back from device to host.
	ret = clEnqueueReadBuffer(command_queue, bufDst, CL_TRUE, 0, sizeof(uint8_t) * width * height * 4 / 8, dst, 0, NULL, NULL);

	// 8. Synchronize and release all OpenCL resources.
	CL_ERR(clFinish(command_queue));
	CL_ERR(clReleaseMemObject(bufSrc));
	CL_ERR(clReleaseMemObject(bufDst));
	CL_ERR(clReleaseCommandQueue(command_queue));
	CL_ERR(clReleaseContext(context));
	
	return 0;
}
