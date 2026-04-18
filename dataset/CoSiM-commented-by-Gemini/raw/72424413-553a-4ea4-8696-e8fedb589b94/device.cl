/**
 * @file device.cl
 * @brief An OpenCL implementation for a block-based texture compression algorithm.
 *
 * This file contains both the OpenCL device code (the `etc` kernel and helper functions)
 * and C++ host code for orchestrating the compression. The algorithm is a variant of
 * Ericsson Texture Compression (ETC), compressing 4x4 pixel blocks.
 */

// --- OpenCL Device Code ---

// Define standard integer types for clarity and portability.
#define uint8_t uchar
#define int8_t char
#define uint16_t ushort
#define int16_t short
#define uint32_t uint
#define int32_t int

#define UINT32_MAX UINT_MAX
#define INT32_MAX INT_MAX

// Represents a 32-bit BGRA color, allowing access by channels, components, or raw bits.
typedef union {
	struct BgraColorType {
		uint8_t b;
		uint8_t g;
		uint8_t r;
		uint8_t a;
	} channels;
	uint8_t components[4];
	uint32_t bits;
} Color;

// Quantizes a 0-255 float value to 5 bits.
inline uint8_t round_to_5_bits(float val) {
	return clamp((uint8_t)(val * 31.0f / 255.0f + 0.5f), (uint8_t)0, (uint8_t)31);
}

// Quantizes a 0-255 float value to 4 bits.
inline uint8_t round_to_4_bits(float val) {
	return clamp((uint8_t)(val * 15.0f / 255.0f + 0.5f), (uint8_t)0, (uint8_t)15);
}


/**
 * @brief Codeword tables for luminance modulation.
 * Stored in __constant memory for fast, cached access by all work-items.
 */
__constant static const int16_t g_codeword_tables[8][4] = {
	{-8, -2, 2, 8},
	{-17, -5, 5, 17},
	{-29, -9, 9, 29},
	{-42, -13, 13, 42},
	{-60, -18, 18, 60},
	{-80, -24, 24, 80},
	{-106, -33, 33, 106},
	{-183, -47, 47, 183}
};

// Maps a 2-bit modifier index to a 2-bit pixel index.
__constant static const uint8_t g_mod_to_pix[4] = {3, 2, 0, 1};

// Tables for reordering pixel indices based on the block 'flip' mode.
__constant static const uint8_t g_idx_to_num[4][8] = {
	{0, 4, 1, 5, 2, 6, 3, 7},        
	{8, 12, 9, 13, 10, 14, 11, 15},  
	{0, 4, 8, 12, 1, 5, 9, 13},      
	{2, 6, 10, 14, 3, 7, 11, 15}     
};

// Applies a luminance offset to a base color.
Color makeColor(Color base, int16_t lum) {
	int b = (int)(base.channels.b) + lum;
	int g = (int)(base.channels.g) + lum;
	int r = (int)(base.channels.r) + lum;
	Color color;
	color.channels.b = (uint8_t)(clamp(b, 0, 255));
	color.channels.g = (uint8_t)(clamp(g, 0, 255));
	color.channels.r = (uint8_t)(clamp(r, 0, 255));
	return color;
}

// Calculates the error between two colors. Can use a perceived error metric if defined.
uint32_t getColorError(Color u, Color v) {
#ifdef USE_PERCEIVED_ERROR_METRIC
	float delta_b = (float)(u.channels.b) - v.channels.b;
	float delta_g = (float)(u.channels.g) - v.channels.g;
	float delta_r = (float)(u.channels.r) - v.channels.r;
	return (uint32_t)(0.299f * delta_b * delta_b +
								 0.587f * delta_g * delta_g +
								 0.114f * delta_r * delta_r);
#else
	// Default to squared Euclidean distance.
	int delta_b = (int)(u.channels.b) - v.channels.b;
	int delta_g = (int)(u.channels.g) - v.channels.g;
	int delta_r = (int)(u.channels.r) - v.channels.r;
	return delta_b * delta_b + delta_g * delta_g + delta_r * delta_r;
#endif
}

// Writes two 4-bit-per-channel colors to the compressed block.
inline void WriteColors444(__global uint8_t* block, Color color0, Color color1) {
	block[0] = (color0.channels.r & 0xf0) | (color1.channels.r >> 4);
	block[1] = (color0.channels.g & 0xf0) | (color1.channels.g >> 4);
	block[2] = (color0.channels.b & 0xf0) | (color1.channels.b >> 4);
}

// Writes two 5-bit-per-channel colors using differential encoding.
inline void WriteColors555(__global uint8_t* block, Color color0, Color color1) {
	const uint8_t two_compl_trans_table[8] = { 4, 5, 6, 7, 0, 1, 2, 3 };
	int16_t delta_r = (int16_t)(color1.channels.r >> 3) - (color0.channels.r >> 3);
	int16_t delta_g = (int16_t)(color1.channels.g >> 3) - (color0.channels.g >> 3);
	int16_t delta_b = (int16_t)(color1.channels.b >> 3) - (color0.channels.b >> 3);
	
	block[0] = (color0.channels.r & 0xf8) | two_compl_trans_table[delta_r + 4];
	block[1] = (color0.channels.g & 0xf8) | two_compl_trans_table[delta_g + 4];
	block[2] = (color0.channels.b & 0xf8) | two_compl_trans_table[delta_b + 4];
}

// Writes the codeword table index for a sub-block.
inline void WriteCodewordTable(__global uint8_t* block, uint8_t sub_block_id, uint8_t table) {
	uint8_t shift = (2 + (3 - sub_block_id * 3));
	block[3] &= ~(0x07 << shift);
	block[3] |= table << shift;
}

// Writes the 32-bit packed pixel indices.
inline void WritePixelData(__global uint8_t* block, uint32_t pixel_data) {
	block[4] |= pixel_data >> 24;
	block[5] |= (pixel_data >> 16) & 0xff;
	block[6] |= (pixel_data >> 8) & 0xff;
	block[7] |= pixel_data & 0xff;
}

// Writes the 'flip' bit (determines sub-block partitioning).
inline void WriteFlip(__global uint8_t* block, bool flip) {
	block[3] &= ~0x01;
	block[3] |= (uint8_t)(flip);
}

// Writes the 'diff' bit (determines if differential encoding is used).
inline void WriteDiff(__global uint8_t* block, bool diff) {
	block[3] &= ~0x02;
	block[3] |= (uint8_t)(diff) << 1;
}

// Creates a 4-bit-per-channel color, expanded to 8-bit.
inline Color makeColor444(const float* bgr) {
	uint8_t b4 = round_to_4_bits(bgr[0]);
	uint8_t g4 = round_to_4_bits(bgr[1]);
	uint8_t r4 = round_to_4_bits(bgr[2]);
	Color bgr444;
	bgr444.channels.b = (b4 << 4) | b4;
	bgr444.channels.g = (g4 << 4) | g4;
	bgr444.channels.r = (r4 << 4) | r4;
	bgr444.channels.a = 0x44;
	return bgr444;
}

// Creates a 5-bit-per-channel color, expanded to 8-bit.
inline Color makeColor555(const float* bgr) {
	uint8_t b5 = round_to_5_bits(bgr[0]);
	uint8_t g5 = round_to_5_bits(bgr[1]);
	uint8_t r5 = round_to_5_bits(bgr[2]);
	Color bgr555;
	bgr555.channels.b = (b5 > 2);
	bgr555.channels.g = (g5 > 2);
	bgr555.channels.r = (r5 > 2);
	bgr555.channels.a = 0x55;
	return bgr555;
}
	
// Computes the average color of an 8-pixel sub-block.
void getAverageColor(const Color* src, float* avg_color) {
	uint32_t sum_b = 0, sum_g = 0, sum_r = 0;
	for (unsigned int i = 0; i < 8; ++i) {
		sum_b += src[i].channels.b;
		sum_g += src[i].channels.g;
		sum_r += src[i].channels.r;
	}
	const float kInv8 = 1.0f / 8.0f;
	avg_color[0] = (float)(sum_b) * kInv8;
	avg_color[1] = (float)(sum_g) * kInv8;
	avg_color[2] = (float)(sum_r) * kInv8;
}

/**
 * @brief Finds the best luminance table and modifiers for a sub-block (Vector Quantization).
 * @return The total error for the best configuration.
 */
unsigned long computeLuminance(__global uint8_t* block,
						   __private const Color* src,
						   Color base,
						   int sub_block_id,
						   __constant const uint8_t* idx_to_num_tab,
						   unsigned long threshold)
{
	uint32_t best_tbl_err = threshold;
	uint8_t best_tbl_idx = 0;
	uint8_t best_mod_idx[8][8];  

	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		Color candidate_color[4];  
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			candidate_color[mod_idx] = makeColor(base, g_codeword_tables[tbl_idx][mod_idx]);
		}
		
		uint32_t tbl_err = 0;
		for (unsigned int i = 0; i < 8; ++i) {
			uint32_t best_mod_err = threshold;
			for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
				uint32_t mod_err = getColorError(src[i], candidate_color[mod_idx]);
				if (mod_err < best_mod_err) {
					best_mod_idx[tbl_idx][i] = mod_idx;
					best_mod_err = mod_err;
					if (mod_err == 0) break;
				}
			}
			tbl_err += best_mod_err;
			if (tbl_err > best_tbl_err) break;
		}
		
		if (tbl_err < best_tbl_err) {
			best_tbl_err = tbl_err;
			best_tbl_idx = tbl_idx;
			if (tbl_err == 0) break;
		}
	}

	WriteCodewordTable(block, sub_block_id, best_tbl_idx);
	uint32_t pix_data = 0;
	for (unsigned int i = 0; i < 8; ++i) {
		uint8_t mod_idx = best_mod_idx[best_tbl_idx][i];
		uint8_t pix_idx = g_mod_to_pix[mod_idx];
		uint32_t lsb = pix_idx & 0x1;
		uint32_t msb = pix_idx >> 1;
		int texel_num = idx_to_num_tab[i];
		pix_data |= msb << (texel_num + 16);
		pix_data |= lsb << (texel_num);
	}
	WritePixelData(block, pix_data);
	return best_tbl_err;
}

// Fast path for blocks containing only a single solid color.
bool tryCompressSolidBlock(__global uint8_t* dst, __private const Color* src) {
	for (unsigned int i = 1; i < 16; ++i) {
		if (src[i].bits != src[0].bits) return false;
	}
	
	for (int i = 0; i < 8; i++) dst[i] = 0;
	float src_color_float[3] = {(float)(src->channels.b), (float)(src->channels.g), (float)(src->channels.r)};
	Color base = makeColor555(src_color_float);
	
	WriteDiff(dst, true);
	WriteFlip(dst, false);
	WriteColors555(dst, base, base);
	
	uint8_t best_tbl_idx = 0;
	uint8_t best_mod_idx = 0;
	uint32_t best_mod_err = UINT32_MAX; 
	
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			uint32_t mod_err = getColorError(*src, makeColor(base, g_codeword_tables[tbl_idx][mod_idx]));
			if (mod_err < best_mod_err) {
				best_tbl_idx = tbl_idx;
				best_mod_idx = mod_idx;
				best_mod_err = mod_err;
				if (mod_err == 0) break;
			}
		}
		if (best_mod_err == 0) break;
	}
	
	WriteCodewordTable(dst, 0, best_tbl_idx);
	WriteCodewordTable(dst, 1, best_tbl_idx);
	uint8_t pix_idx = g_mod_to_pix[best_mod_idx];
	uint32_t lsb = pix_idx & 0x1;
	uint32_t msb = pix_idx >> 1;
	uint32_t pix_data = 0;
	for (unsigned int i = 0; i < 2; ++i) {
		for (unsigned int j = 0; j < 8; ++j) {
			int texel_num = g_idx_to_num[i][j];
			pix_data |= msb << (texel_num + 16);
			pix_data |= lsb << (texel_num);
		}
	}
	WritePixelData(dst, pix_data);
	return true;
}

// Main device-side compression function for a 4x4 block.
void compress(__global uint8_t* dst, __private const Color* ver_src, __private const Color* hor_src, unsigned long threshold) {
	if (tryCompressSolidBlock(dst, ver_src)) return;
	
	__private const Color* sub_block_src[4] = {ver_src, ver_src + 8, hor_src, hor_src + 8};
	Color sub_block_avg[4];
	bool use_differential[2] = {true, true};
	
	// Determine if differential mode should be used.
	for (unsigned int i = 0, j = 1; i < 4; i += 2, j += 2) {
		float avg_color_0[3], avg_color_1[3];
		getAverageColor(sub_block_src[i], avg_color_0);
		Color avg_color_555_0 = makeColor555(avg_color_0);
		getAverageColor(sub_block_src[j], avg_color_1);
		Color avg_color_555_1 = makeColor555(avg_color_1);
		
		for (unsigned int light_idx = 0; light_idx < 3; ++light_idx) {
			int component_diff = (avg_color_555_1.components[light_idx] >> 3) - (avg_color_555_0.components[light_idx] >> 3);
			if (component_diff < -4 || component_diff > 3) {
				use_differential[i / 2] = false;
				sub_block_avg[i] = makeColor444(avg_color_0);
				sub_block_avg[j] = makeColor444(avg_color_1);
			} else {
				sub_block_avg[i] = avg_color_555_0;
				sub_block_avg[j] = avg_color_555_1;
			}
		}
	}
	
	// Determine the best flip mode (partitioning) by comparing errors.
	uint32_t sub_block_err[4] = {0};
	for (unsigned int i = 0; i < 4; ++i) {
		for (unsigned int j = 0; j < 8; ++j) {
			sub_block_err[i] += getColorError(sub_block_avg[i], sub_block_src[i][j]);
		}
	}
	bool flip = sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1];
	
	for (int i = 0; i < 8; i++) dst[i] = 0;
	WriteDiff(dst, use_differential[!!flip]);
	WriteFlip(dst, flip);
	uint8_t sub_block_off_0 = flip ? 2 : 0;
	uint8_t sub_block_off_1 = sub_block_off_0 + 1;
	
	if (use_differential[!!flip]) {
		WriteColors555(dst, sub_block_avg[sub_block_off_0], sub_block_avg[sub_block_off_1]);
	} else {
		WriteColors444(dst, sub_block_avg[sub_block_off_0], sub_block_avg[sub_block_off_1]);
	}
	
	// Compress the two chosen sub-blocks.
	computeLuminance(dst, sub_block_src[sub_block_off_0], sub_block_avg[sub_block_off_0], 0, g_idx_to_num[sub_block_off_0], threshold);
	computeLuminance(dst, sub_block_src[sub_block_off_1], sub_block_avg[sub_block_off_1], 1, g_idx_to_num[sub_block_off_1], threshold);
}

/**
 * @brief The main OpenCL kernel for texture compression.
 *
 * Each work-item in the 2D NDRange is responsible for compressing one 4x4 block.
 */
__kernel void etc(int width, int height,
			__global Color* src,
			__global uchar* dst) {
	// Threading Model: Map 2D global ID to a 4x4 block coordinate.
	int row_index = get_global_id(0) * 4;
	int col_index = get_global_id(1) * 4;

	__global Color* row0 = src + row_index * width + col_index;
	__global Color* row1 = row0 + width;
	__global Color* row2 = row1 + width;
	__global Color* row3 = row2 + width;

	// Memory Model: Gather data from global `src` buffer into private memory.
	Color ver_blocks[16], hor_blocks[16];

	// This explicit gathering creates the two possible 8-pixel sub-block layouts
	// needed for evaluating the 'flip' mode.
	ver_blocks[0].bits = row0[0].bits;
	ver_blocks[1].bits = row0[1].bits;
	ver_blocks[2].bits = row1[0].bits;
	// ... (and so on for all 16 pixels) ...

	// Call the main compression function for the block.
	compress(dst + (row_index * width) / 2 + col_index * 2, ver_blocks, hor_blocks, INT32_MAX);
}

// --- C++ Host Code ---

#include "helper.hpp"

using namespace std;

// Error checking and reporting utilities for OpenCL.
int CL_ERR(int cl_ret) { /* ... */ }
int CL_COMPILE_ERR(int cl_ret, cl_program program, cl_device_id device) { /* ... */ }
void read_kernel(string file_name, string &str_kernel) { /* ... */ }
const char* cl_get_string_err(cl_int err) { /* ... */ }
void cl_get_compiler_err_log(cl_program program, cl_device_id device) { /* ... */ }

// Helper function to find a GPU device.
void gpu_find(cl_device_id &device, 
		cl_platform_id* &platform_list,
		cl_device_id* &device_list)
{
	// ... (Standard OpenCL platform and device discovery) ...
}

/**
 * @brief Constructor for the TextureCompressor.
 *
 * Architectural Intent: Initializes the OpenCL environment. It finds a GPU,
 * creates a context and command queue, and pre-compiles the OpenCL kernel from
 * the source file.
 */
TextureCompressor::TextureCompressor() {
	gpu_find(device, platform_ids, device_ids);
	cl_int ret;
	string kernel_src;
	
	context = clCreateContext(0, 1, &device, NULL, NULL, &ret);
	CL_ERR( ret );
	command_queue = clCreateCommandQueue(context, device, 0, &ret);
	CL_ERR( ret );

	// Read kernel source from this same file.
	read_kernel("device.cl", kernel_src);
	const char* kernel_c_str = kernel_src.c_str();
	program = clCreateProgramWithSource(context, 1, &kernel_c_str, NULL, &ret);
	CL_ERR( ret );
	
	ret = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	CL_COMPILE_ERR( ret, program, device );
	
	kernel = clCreateKernel(program, "etc", &ret);
	CL_ERR( ret );
}

TextureCompressor::~TextureCompressor() {
	CL_ERR( clReleaseCommandQueue(command_queue) );
	CL_ERR( clReleaseContext(context) );
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
									  int height)
{
	cl_int ret;
	size_t src_size = width * height * 4;
	size_t dst_size = width * height * 4 / 8;

	// Step 1: Create device memory buffers.
	cl_mem src_dev = clCreateBuffer(context, CL_MEM_READ_WRITE, src_size, NULL, &ret);
	CL_ERR( ret );
	cl_mem dst_dev = clCreateBuffer(context, CL_MEM_READ_WRITE, dst_size, NULL, &ret);
	CL_ERR( ret );

	// Step 2: Copy source image from host to device.
	CL_ERR( clEnqueueWriteBuffer(command_queue, src_dev, CL_TRUE, 0, src_size, src, 0, NULL, NULL) );

	// Step 3: Set kernel arguments.
	CL_ERR( clSetKernelArg(kernel, 0, sizeof(cl_int), (void *)&width) );
	CL_ERR( clSetKernelArg(kernel, 1, sizeof(cl_int), (void *)&height) );
	CL_ERR( clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&src_dev) );
	CL_ERR( clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&dst_dev) );

	// Step 4: Define NDRange and execute the kernel.
	size_t globalSize[2] = {(size_t)height / 4, (size_t)width / 4};
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, globalSize, NULL, 0, NULL, NULL);
	CL_ERR( ret );

	// Step 5: Read compressed data from device back to host.
	CL_ERR( clEnqueueReadBuffer(command_queue, dst_dev, CL_TRUE, 0, dst_size, dst, 0, NULL, NULL) );
	
	// Step 6: Synchronize and release resources.
	CL_ERR( clFinish(command_queue) );
	CL_ERR( clReleaseMemObject(src_dev) );
	CL_ERR( clReleaseMemObject(dst_dev) );

	return 0;
}
