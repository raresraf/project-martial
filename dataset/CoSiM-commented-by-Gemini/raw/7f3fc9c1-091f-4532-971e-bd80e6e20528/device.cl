/**
 * @file device.cl
 * @brief Implements an ETC-like texture compression algorithm.
 *
 * This file contains a mix of C++ and OpenCL C code for texture compression.
 * The primary functionality is encapsulated within the TextureCompressor class, which
 * performs the compression. The OpenCL kernel `foo` appears to be a placeholder
 * or for testing purposes. The C++ code includes logic for color manipulation,
 * error calculation, and writing compressed block data.
 */

// A simple OpenCL kernel, likely for testing or as a placeholder.
__kernel void foo(cl_int width,
        cl_int height,
        __global cl_uchar* src,
        __global cl_uchar* dst)
{
	dst[0] = 2;
	dst[1] = width;
	dst[2] = height; 
        uint k;
}

#include "compress.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>
#include <cstdint>

using namespace std;

// Clamps a value between a minimum and maximum.
template 
inline T clamp(T val, T min, T max) {
	return val < min ? min : (val > max ? max : val);
}

// Rounds a float color component (0-255) to a 5-bit value.
inline uint8_t round_to_5_bits(float val) {
	return clamp(val * 31.0f / 255.0f + 0.5f, 0.0f, 31.0f);
}

// Rounds a float color component (0-255) to a 4-bit value.
inline uint8_t round_to_4_bits(float val) {
	return clamp(val * 15.0f / 255.0f + 0.5f, 0.0f, 15.0f);
}

// Reads an OpenCL kernel source file into a string.
void read_kernel(string file_name, string &str_kernel)
{
	ifstream in_file(file_name.c_str());
	if (!in_file.is_open()) {
		return;
	}
	stringstream str_stream;
	str_stream << in_file.rdbuf();
	str_kernel = str_stream.str();
}

// Codeword tables for luminance modification in ETC compression.
ALIGNAS(16) static const int16_t g_codeword_tables[8][4] = {
	{-8, -2, 2, 8},
	{-17, -5, 5, 17},
	{-29, -9, 9, 29},
	{-42, -13, 13, 42},
	{-60, -18, 18, 60},
	{-80, -24, 24, 80},
	{-106, -33, 33, 106},
	{-183, -47, 47, 183}};

// Maps a modifier index to a pixel index.
static const uint8_t g_mod_to_pix[4] = {3, 2, 0, 1};

// Tables for reordering pixel indices.
static const uint8_t g_idx_to_num[4][8] = {
	{0, 4, 1, 5, 2, 6, 3, 7},        
	{8, 12, 9, 13, 10, 14, 11, 15},  
	{0, 4, 8, 12, 1, 5, 9, 13},      
	{2, 6, 10, 14, 3, 7, 11, 15}     
};

// Creates a color by adding a luminance value to a base color.
inline Color makeColor(const Color& base, int16_t lum) {
	int b = static_cast<int>(base.channels.b) + lum;
	int g = static_cast<int>(base.channels.g) + lum;
	int r = static_cast<int>(base.channels.r) + lum;
	Color color;
	color.channels.b = static_cast<uint8_t>(clamp(b, 0, 255));
	color.channels.g = static_cast<uint8_t>(clamp(g, 0, 255));
	color.channels.r = static_cast<uint8_t>(clamp(r, 0, 255));
	return color;
}

// Calculates the squared error between two colors.
inline uint32_t getColorError(const Color& u, const Color& v) {
#ifdef USE_PERCEIVED_ERROR_METRIC
	float delta_b = static_cast<float>(u.channels.b) - v.channels.b;
	float delta_g = static_cast<float>(u.channels.g) - v.channels.g;
	float delta_r = static_cast<float>(u.channels.r) - v.channels.r;
	return static_cast<uint32_t>(0.299f * delta_r * delta_r +
								 0.587f * delta_g * delta_g +
								 0.114f * delta_b * delta_b);
#else
	int delta_b = static_cast<int>(u.channels.b) - v.channels.b;
	int delta_g = static_cast<int>(u.channels.g) - v.channels.g;
	int delta_r = static_cast<int>(u.channels.r) - v.channels.r;
	return delta_b * delta_b + delta_g * delta_g + delta_r * delta_r;
#endif
}

// Writes two 4-bit colors to a compressed block.
inline void WriteColors444(uint8_t* block,
						   const Color& color0,
						   const Color& color1) {
	block[0] = (color0.channels.r & 0xf0) | (color1.channels.r >> 4);
	block[1] = (color0.channels.g & 0xf0) | (color1.channels.g >> 4);
	block[2] = (color0.channels.b & 0xf0) | (color1.channels.b >> 4);
}

// Writes two 5-bit colors to a compressed block with differential encoding.
inline void WriteColors555(uint8_t* block,
						   const Color& color0,
						   const Color& color1) {
	static const uint8_t two_compl_trans_table[8] = { 4, 5, 6, 7, 0, 1, 2, 3 };
	
	int16_t delta_r = static_cast<int16_t>(color1.channels.r >> 3) - (color0.channels.r >> 3);
	int16_t delta_g = static_cast<int16_t>(color1.channels.g >> 3) - (color0.channels.g >> 3);
	int16_t delta_b = static_cast<int16_t>(color1.channels.b >> 3) - (color0.channels.b >> 3);
	
	block[0] = (color0.channels.r & 0xf8) | two_compl_trans_table[delta_r + 4];
	block[1] = (color0.channels.g & 0xf8) | two_compl_trans_table[delta_g + 4];
	block[2] = (color0.channels.b & 0xf8) | two_compl_trans_table[delta_b + 4];
}

// Writes the codeword table index to the compressed block.
inline void WriteCodewordTable(uint8_t* block,
							   uint8_t sub_block_id,
							   uint8_t table) {
	uint8_t shift = (2 + (3 - sub_block_id) * 3);
	block[3] &= ~(0x07 << shift);
	block[3] |= table << shift;
}

// Writes the 32-bit pixel data to the compressed block.
inline void WritePixelData(uint8_t* block, uint32_t pixel_data) {
	block[4] = pixel_data & 0xff;
	block[5] = (pixel_data >> 8) & 0xff;
	block[6] = (pixel_data >> 16) & 0xff;
	block[7] = (pixel_data >> 24) & 0xff;
}

// Writes the flip bit to the compressed block.
inline void WriteFlip(uint8_t* block, bool flip) {
	block[3] &= ~0x01;
	block[3] |= static_cast<uint8_t>(flip);
}

// Writes the differential bit to the compressed block.
inline void WriteDiff(uint8_t* block, bool diff) {
	block[3] &= ~0x02;
	block[3] |= static_cast<uint8_t>(diff) << 1;
}

// Extracts a 4x4 block of pixels from a source image.
inline void ExtractBlock(uint8_t* dst, const uint8_t* src, int width) {
	for (int j = 0; j < 4; ++j) {
		memcpy(&dst[j * 16], src, 16);
		src += width * 4;
	}
}

// Creates a 4:4:4 color from a float color.
inline Color makeColor444(const float* bgr) {
	uint8_t b4 = round_to_4_bits(bgr[0]);
	uint8_t g4 = round_to_4_bits(bgr[1]);
	uint8_t r4 = round_to_4_bits(bgr[2]);
	Color bgr444;
	bgr444.channels.b = (b4 << 4) | b4;
	bgr444.channels.g = (g4 << 4) | g4;
	bgr444.channels.r = (r4 << 4) | r4;
	bgr444.channels.a = 0;
	return bgr444;
}

// Creates a 5:5:5 color from a float color.
inline Color makeColor555(const float* bgr) {
	uint8_t b5 = round_to_5_bits(bgr[0]);
	uint8_t g5 = round_to_5_bits(bgr[1]);
	uint8_t r5 = round_to_5_bits(bgr[2]);
	Color bgr555;
	bgr555.channels.b = (b5 << 3) | (b5 >> 2);
	bgr555.channels.g = (g5 << 3) | (g5 >> 2);
	bgr555.channels.r = (r5 << 3) | (r5 >> 2);
	bgr555.channels.a = 0;
	return bgr555;
}

// Calculates the average color of an 8-pixel sub-block.
void getAverageColor(const Color* src, float* avg_color)
{
	uint32_t sum_b = 0, sum_g = 0, sum_r = 0;
	for (unsigned int i = 0; i < 8; ++i) {
		sum_b += src[i].channels.b;
		sum_g += src[i].channels.g;
		sum_r += src[i].channels.r;
	}
	
	const float kInv8 = 1.0f / 8.0f;
	avg_color[0] = static_cast<float>(sum_b) * kInv8;
	avg_color[1] = static_cast<float>(sum_g) * kInv8;
	avg_color[2] = static_cast<float>(sum_r) * kInv8;
}

// Computes the best luminance table and modifier for a sub-block to minimize error.
unsigned long computeLuminance(uint8_t* block,
						   const Color* src,
						   const Color& base,
						   int sub_block_id,
						   const uint8_t* idx_to_num_tab,
						   unsigned long threshold)
{
	uint32_t best_tbl_err = threshold;
	uint8_t best_tbl_idx = 0;
	uint8_t best_mod_idx[8][8];

	// Iterate through all possible codeword tables.
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		// Generate candidate colors based on the current table.
		Color candidate_color[4];
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			int16_t lum = g_codeword_tables[tbl_idx][mod_idx];
			candidate_color[mod_idx] = makeColor(base, lum);
		}
		
		uint32_t tbl_err = 0;
		// For each pixel in the sub-block, find the best modifier.
		for (unsigned int i = 0; i < 8; ++i) {
			uint32_t best_mod_err = threshold;
			for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
				const Color& color = candidate_color[mod_idx];
				uint32_t mod_err = getColorError(src[i], color);
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
	// Write the pixel data based on the best modifiers found.
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

// Attempts to compress a block as a solid color if all pixels are the same.
bool tryCompressSolidBlock(uint8_t* dst,
						   const Color* src,
						   unsigned long* error)
{
	for (unsigned int i = 1; i < 16; ++i) {
		if (src[i].bits != src[0].bits)
			return false;
	}
	
	memset(dst, 0, 8);
	
	float src_color_float[3] = {static_cast<float>(src->channels.b),
		static_cast<float>(src->channels.g),
		static_cast<float>(src->channels.r)};
	Color base = makeColor555(src_color_float);
	
	WriteDiff(dst, true);
	WriteFlip(dst, false);
	WriteColors555(dst, base, base);
	
	uint8_t best_tbl_idx = 0;
	uint8_t best_mod_idx = 0;
	uint32_t best_mod_err = UINT32_MAX; 
	
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			int16_t lum = g_codeword_tables[tbl_idx][mod_idx];
			const Color& color = makeColor(base, lum);
			uint32_t mod_err = getColorError(*src, color);
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
	*error = 16 * best_mod_err;
	return true;
}

/**
 * @brief Constructor for TextureCompressor. Initializes OpenCL and selects a device.
 */
TextureCompressor::TextureCompressor() {
	cl_uint platform_num = 0;
	platform_ids = NULL;
	cl_uint device_num = 0;
	device_ids = NULL;

	clGetPlatformIDs(0, NULL, &platform_num);
	if (platform_num == 0) return;
	platform_ids = new cl_platform_id[platform_num];
	clGetPlatformIDs(platform_num, platform_ids, NULL);

	for(cl_uint platf = 0; platf < platform_num; platf++)
	{
		clGetDeviceIDs(platform_ids[platf], CL_DEVICE_TYPE_ALL, 0, NULL, &device_num);
		if (device_num == 0) continue;

		device_ids = new cl_device_id[device_num];
		clGetDeviceIDs(platform_ids[platf], CL_DEVICE_TYPE_ALL, device_num, device_ids, NULL);
		
		if (device_num > 0) {
			device = device_ids[0]; // Select the first available device
			break;
		}
	}
 }

/**
 * @brief Destructor for TextureCompressor. Releases OpenCL resources.
 */
TextureCompressor::~TextureCompressor() { 
	delete[] platform_ids;
	delete[] device_ids;
}

/**
 * @brief Compresses a single 4x4 block of pixels.
 * @param dst Pointer to the destination buffer for the compressed block.
 * @param ver_src Pointer to the source pixels arranged for vertical sub-blocks.
 * @param hor_src Pointer to the source pixels arranged for horizontal sub-blocks.
 * @param threshold The error threshold for compression.
 * @return The total compression error for the block.
 */
unsigned long TextureCompressor::compressBlock(uint8_t* dst,
												   const Color* ver_src,
												   const Color* hor_src,
												   unsigned long threshold)
{
	unsigned long solid_error = 0;
	if (tryCompressSolidBlock(dst, ver_src, &solid_error)) {
		return solid_error;
	}
	
	const Color* sub_block_src[4] = {ver_src, ver_src + 8, hor_src, hor_src + 8};
	
	Color sub_block_avg[4];
	bool use_differential[2] = {true, true};
	
	for (unsigned int i = 0, j = 1; i < 4; i += 2, j += 2) {
		float avg_color_0[3];
		getAverageColor(sub_block_src[i], avg_color_0);
		Color avg_color_555_0 = makeColor555(avg_color_0);
		
		float avg_color_1[3];
		getAverageColor(sub_block_src[j], avg_color_1);
		Color avg_color_555_1 = makeColor555(avg_color_1);
		
		for (unsigned int light_idx = 0; light_idx < 3; ++light_idx) {
			int u = avg_color_555_0.components[light_idx] >> 3;
			int v = avg_color_555_1.components[light_idx] >> 3;
			
			int component_diff = v - u;
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
	
	uint32_t sub_block_err[4] = {0};
	for (unsigned int i = 0; i < 4; ++i) {
		for (unsigned int j = 0; j < 8; ++j) {
			sub_block_err[i] += getColorError(sub_block_avg[i], sub_block_src[i][j]);
		}
	}
	
	bool flip = sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1];
	
	memset(dst, 0, 8);
	
	WriteDiff(dst, use_differential[!!flip]);
	WriteFlip(dst, flip);
	
	uint8_t sub_block_off_0 = flip ? 2 : 0;
	uint8_t sub_block_off_1 = sub_block_off_0 + 1;
	
	if (use_differential[!!flip]) {
		WriteColors555(dst, sub_block_avg[sub_block_off_0], sub_block_avg[sub_block_off_1]);
	} else {
		WriteColors444(dst, sub_block_avg[sub_block_off_0], sub_block_avg[sub_block_off_1]);
	}
	
	unsigned long lumi_error1 = 0, lumi_error2 = 0;
	
	lumi_error1 = computeLuminance(dst, sub_block_src[sub_block_off_0],
								   sub_block_avg[sub_block_off_0], 0,
								   g_idx_to_num[sub_block_off_0],
								   threshold);
	
	lumi_error2 = computeLuminance(dst, sub_block_src[sub_block_off_1],
								   sub_block_avg[sub_block_off_1], 1,
								   g_idx_to_num[sub_block_off_1],
								   threshold);
	
	return lumi_error1 + lumi_error2;
}

/**
 * @brief Compresses a texture.
 * @param src Pointer to the source texture data.
 * @param dst Pointer to the destination buffer for the compressed data.
 * @param width The width of the texture.
 * @param height The height of the texture.
 * @return The total compression error.
 */
unsigned long TextureCompressor::compress(const uint8_t* src,
											  uint8_t* dst,
											  int width,
											  int height) {
	Color ver_blocks[16];
	Color hor_blocks[16];
	string kernel_src;
	cl_int ret;

	context = clCreateContext(0, 1, &device, NULL, NULL, &ret);
	command_queue = clCreateCommandQueue(context, device, 0, &ret);
	
	cl_mem src_gpu = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				  width * height * 4, (void*)src, &ret);
	cl_mem dst_gpu = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
				  width * height / 2, NULL, &ret);
	
	read_kernel("device.cl", kernel_src);
	const char* kernel_c_str = kernel_src.c_str();
	
	program = clCreateProgramWithSource(context, 1, &kernel_c_str, NULL, &ret);
	clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	kernel = clCreateKernel(program, "foo", &ret);
	
	clSetKernelArg(kernel, 0, sizeof(cl_int), (void *)&width);
	clSetKernelArg(kernel, 1, sizeof(cl_int), (void *)&height);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&src_gpu);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&dst_gpu);
	
	size_t globalSize[2] = {(size_t)(width / 4), (size_t)(height / 4)};
	clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, globalSize, NULL, 0, NULL, NULL);
	
	clEnqueueReadBuffer(command_queue, dst_gpu, CL_TRUE, 0, width * height / 2, dst, 0, NULL, NULL);
	
	unsigned long compressed_error = 0;
	// This host-side loop seems redundant if the kernel is meant to do the compression.
	// It might be a fallback or part of a hybrid approach.
	for (int y = 0; y < height; y += 4, src += width * 4 * 4) {
		for (int x = 0; x < width; x += 4, dst += 8) {
			const Color* row0 = reinterpret_cast<const Color*>(src + x * 4);
			const Color* row1 = row0 + width;
			const Color* row2 = row1 + width;
			const Color* row3 = row2 + width;
			
			// Re-arrange pixels for vertical and horizontal sub-blocks.
			memcpy(ver_blocks, row0, 8);
			memcpy(ver_blocks + 2, row1, 8);
			memcpy(ver_blocks + 4, row2, 8);
			memcpy(ver_blocks + 6, row3, 8);
			memcpy(ver_blocks + 8, row0 + 2, 8);
			memcpy(ver_blocks + 10, row1 + 2, 8);
			memcpy(ver_blocks + 12, row2 + 2, 8);
			memcpy(ver_blocks + 14, row3 + 2, 8);
			
			memcpy(hor_blocks, row0, 16);
			memcpy(hor_blocks + 4, row1, 16);
			memcpy(hor_blocks + 8, row2, 16);
			memcpy(hor_blocks + 12, row3, 16);
			
			compressed_error += compressBlock(dst, ver_blocks, hor_blocks, UINT32_MAX);
		}
	}

	clReleaseMemObject(src_gpu);
	clReleaseMemObject(dst_gpu);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);

	return compressed_error;
}
