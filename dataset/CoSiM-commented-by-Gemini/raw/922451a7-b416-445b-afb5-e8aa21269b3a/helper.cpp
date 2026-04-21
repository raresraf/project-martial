/**
 * @file helper.cpp
 * @brief OpenCL utility set and ETC1 texture compression implementation.
 * 
 * This module provides essential utilities for OpenCL host applications,
 * including error management, device discovery, and kernel loading. It also
 * features a complete implementation of the ETC1 compression algorithm for 
 * both host and device-side execution.
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "helper.hpp"

using namespace std;

/**
 * @brief Checks and reports OpenCL runtime errors.
 * 
 * Reports the error string associated with a non-zero return code.
 */
int CL_ERR(int cl_ret) {
	if(cl_ret != CL_SUCCESS){
		cout << endl << cl_get_string_err(cl_ret) << endl;
		return 1;
	}
	return 0;
}

/**
 * @brief Checks and reports OpenCL kernel compilation errors.
 * 
 * Additionally retrieves and prints the program build log for diagnostics.
 */
int CL_COMPILE_ERR(int cl_ret, cl_program program, cl_device_id device) {
	if(cl_ret != CL_SUCCESS){
		cout << endl << cl_get_string_err(cl_ret) << endl;
		cl_get_compiler_err_log(program, device);
		return 1;
	}
	return 0;
}

/**
 * @brief Reads OpenCL kernel source from a file.
 */
void read_kernel(string file_name, string &str_kernel) {
	ifstream in_file(file_name.c_str());
	DIE( !in_file.is_open(), "ERR OpenCL kernel file. Same directory as binary ?" );
	stringstream str_stream;
	str_stream << in_file.rdbuf();
	str_kernel = str_stream.str();
}

/**
 * @brief Maps OpenCL error codes to human-readable strings.
 */
const char* cl_get_string_err(cl_int err) {
switch (err) {
  case CL_SUCCESS: return "Success!";
  case CL_DEVICE_NOT_FOUND: return "Device not found.";
  case CL_DEVICE_NOT_AVAILABLE: return "Device not available";
  case CL_COMPILER_NOT_AVAILABLE: return "Compiler not available";
  case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "Memory object alloc fail";
  case CL_OUT_OF_RESOURCES: return "Out of resources";
  case CL_OUT_OF_HOST_MEMORY: return "Out of host memory";
  case CL_PROFILING_INFO_NOT_AVAILABLE: return "Profiling information N/A";
  case CL_MEM_COPY_OVERLAP: return "Memory copy overlap";
  case CL_IMAGE_FORMAT_MISMATCH: return "Image format mismatch";
  case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "Image format no support";
  case CL_BUILD_PROGRAM_FAILURE: return "Program build failure";
  case CL_MAP_FAILURE: return "Map failure";
  case CL_INVALID_VALUE: return "Invalid value";
  case CL_INVALID_DEVICE_TYPE: return "Invalid device type";
  case CL_INVALID_PLATFORM: return "Invalid platform";
  case CL_INVALID_DEVICE: return "Invalid device";
  case CL_INVALID_CONTEXT: return "Invalid context";
  case CL_INVALID_QUEUE_PROPERTIES: return "Invalid queue properties";
  case CL_INVALID_COMMAND_QUEUE: return "Invalid command queue";
  case CL_INVALID_HOST_PTR: return "Invalid host pointer";
  case CL_INVALID_MEM_OBJECT: return "Invalid memory object";
  case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "Invalid image format desc";
  case CL_INVALID_IMAGE_SIZE: return "Invalid image size";
  case CL_INVALID_SAMPLER: return "Invalid sampler";
  case CL_INVALID_BINARY: return "Invalid binary";
  case CL_INVALID_BUILD_OPTIONS: return "Invalid build options";
  case CL_INVALID_PROGRAM: return "Invalid program";
  case CL_INVALID_PROGRAM_EXECUTABLE: return "Invalid program exec";
  case CL_INVALID_KERNEL_NAME: return "Invalid kernel name";
  case CL_INVALID_KERNEL_DEFINITION: return "Invalid kernel definition";
  case CL_INVALID_KERNEL: return "Invalid kernel";
  case CL_INVALID_ARG_INDEX: return "Invalid argument index";
  case CL_INVALID_ARG_VALUE: return "Invalid argument value";
  case CL_INVALID_ARG_SIZE: return "Invalid argument size";
  case CL_INVALID_KERNEL_ARGS: return "Invalid kernel arguments";
  case CL_INVALID_WORK_DIMENSION: return "Invalid work dimension";
  case CL_INVALID_WORK_GROUP_SIZE: return "Invalid work group size";
  case CL_INVALID_WORK_ITEM_SIZE: return "Invalid work item size";
  case CL_INVALID_GLOBAL_OFFSET: return "Invalid global offset";
  case CL_INVALID_EVENT_WAIT_LIST: return "Invalid event wait list";
  case CL_INVALID_EVENT: return "Invalid event";
  case CL_INVALID_OPERATION: return "Invalid operation";
  case CL_INVALID_GL_OBJECT: return "Invalid OpenGL object";
  case CL_INVALID_BUFFER_SIZE: return "Invalid buffer size";
  case CL_INVALID_MIP_LEVEL: return "Invalid mip-map level";
  default: return "Unknown";
  }
}

/**
 * @brief Retrieves the compilation log for an OpenCL program.
 */
void cl_get_compiler_err_log(cl_program program, cl_device_id device) {
	char* build_log;
	size_t log_size;
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
	build_log = new char[ log_size + 1 ];
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
	build_log[ log_size ] = '\0';
	cout << endl << build_log << endl;
	delete[] build_log;
}

>>>> file: helper.hpp
#ifndef CL_HELPER_H
#define CL_HELPER_H

#if __APPLE__
   #include <OpenCL/opencl.h>
#else
   #include <CL/cl.h>
#endif

#include <iostream>

using namespace std;

/**
 * @brief OpenCL error checking.
 */
int CL_ERR(int cl_ret);

/**
 * @brief OpenCL build error checking.
 */
int CL_COMPILE_ERR(int cl_ret, cl_program program, cl_device_id device);

/**
 * @brief Translate OpenCL error codes.
 */
const char* cl_get_string_err(cl_int err);

/**
 * @brief Get program build log.
 */
void cl_get_compiler_err_log(cl_program program, cl_device_id device);

/**
 * @brief Kernel source loader.
 */
void read_kernel(string file_name, string &str_kernel);

/**
 * @brief Fatal error reporting macro.
 */
#define DIE(assertion, call_description)                    \
do {                                                        \
    if (assertion) {                                        \
            fprintf(stderr, "(%d): ",                       \
                            __LINE__);                      \
            perror(call_description);                       \
            exit(EXIT_FAILURE);                             \
    }                                                       \
} while(0);

#endif

>>>> file: texture_compress_device.cl
/**
 * @file texture_compress_device.cl
 * @brief OpenCL Kernels for ETC1 Texture Compression.
 */

/**
 * @brief BGRA color union for pixel manipulation.
 */
typedef union color {
	struct BgraColorType {
		uchar b; /**< Blue channel */
		uchar g; /**< Green channel */
		uchar r; /**< Red channel */
		uchar a; /**< Alpha channel */
	} channels;
	uchar components[4]; /**< Index-based channel access */
	uint bits;           /**< Raw 32-bit access */
} Color;

/**
 * @brief Clamp uchar value to [min, max].
 */
inline uchar uchar_clamp(uchar val, uchar min, uchar max) {
	return val < min ? min : (val > max ? max : val);
}

/**
 * @brief Clamp int value to [min, max].
 */
inline int int_clamp(int val, int min, int max) {
	return val < min ? min : (val > max ? max : val);
}

/**
 * @brief Quantize float to 5-bit uchar.
 */
inline uchar round_to_5_bits(float val) {
	return uchar_clamp(val * 31.0f / 255.0f + 0.5f, 0, 31);
}

/**
 * @brief Quantize float to 4-bit uchar.
 */
inline uchar round_to_4_bits(float val) {
	return uchar_clamp(val * 15.0f / 255.0f + 0.5f, 0, 15);
}

/**
 * @brief ETC1 Codeword tables (Table 3.17.2).
 */
static __constant short g_codeword_tables[8][4] __attribute__((aligned(16))) = {
	{-8, -2, 2, 8}, {-17, -5, 5, 17}, {-29, -9, 9, 29}, {-42, -13, 13, 42},
	{-60, -18, 18, 60}, {-80, -24, 24, 80}, {-106, -33, 33, 106}, {-183, -47, 47, 183}};

/**
 * @brief ETC1 modifier to pixel index mapping (Table 3.17.3).
 */
static __constant uchar g_mod_to_pix[4] = {3, 2, 0, 1};

/**
 * @brief Block indexing for sub-block extraction.
 */
static __constant uchar g_idx_to_num[4][8] = {
	{0, 4, 1, 5, 2, 6, 3, 7}, {8, 12, 9, 13, 10, 14, 11, 15},  
	{0, 4, 8, 12, 1, 5, 9, 13}, {2, 6, 10, 14, 3, 7, 11, 15}};

/**
 * @brief Derives a color with luminance offset.
 */
inline Color makeColor(const Color base, short lum) {
	int b = (int)(base.channels.b) + lum;
	int g = (int)(base.channels.g) + lum;
	int r = (int)(base.channels.r) + lum;
	Color color;
	color.channels.b = (uchar)int_clamp(b, 0, 255);
	color.channels.g = (uchar)int_clamp(g, 0, 255);
	color.channels.r = (uchar)int_clamp(r, 0, 255);
	return color;
}

/**
 * @brief Calculates the error between two colors.
 */
inline uint getColorError(const Color u, const Color v) {
#ifdef USE_PERCEIVED_ERROR_METRIC
	float delta_b = (float)(u.channels.b) - v.channels.b;
	float delta_g = (float)(u.channels.g) - v.channels.g;
	float delta_r = (float)(u.channels.r) - v.channels.r;
	return (uint)(0.299f * delta_b * delta_b + 0.587f * delta_g * delta_g + 0.114f * delta_r * delta_r);
#else
	int delta_b = (int)(u.channels.b) - v.channels.b;
	int delta_g = (int)(u.channels.g) - v.channels.g;
	int delta_r = (int)(u.channels.r) - v.channels.r;
	return delta_b * delta_b + delta_g * delta_g + delta_r * delta_r;
#endif
}

/**
 * @brief Writes 444-format colors to the compressed block.
 */
inline void WriteColors444(__global uchar* block, const Color color0, const Color color1) {
	block[0] = (color0.channels.r & 0xf0) | (color1.channels.r >> 4);
	block[1] = (color0.channels.g & 0xf0) | (color1.channels.g >> 4);
	block[2] = (color0.channels.b & 0xf0) | (color1.channels.b >> 4);
}

/**
 * @brief Writes 555-format differential colors to the compressed block.
 */
inline void WriteColors555(__global uchar* block, const Color color0, const Color color1) {
	const uchar two_compl_trans_table[8] = {4, 5, 6, 7, 0, 1, 2, 3};
	short delta_r = (short)(color1.channels.r >> 3) - (color0.channels.r >> 3);
	short delta_g = (short)(color1.channels.g >> 3) - (color0.channels.g >> 3);
	short delta_b = (short)(color1.channels.b >> 3) - (color0.channels.b >> 3);
	block[0] = (color0.channels.r & 0xf8) | two_compl_trans_table[delta_r + 4];
	block[1] = (color0.channels.g & 0xf8) | two_compl_trans_table[delta_g + 4];
	block[2] = (color0.channels.b & 0xf8) | two_compl_trans_table[delta_b + 4];
}

/**
 * @brief Encodes the codeword table index.
 */
inline void WriteCodewordTable(__global uchar* block, uchar sub_block_id, uchar table) {
	uchar shift = (2 + (3 - sub_block_id * 3));
	block[3] &= ~(0x07 << shift);
	block[3] |= table << shift;
}

/**
 * @brief Encodes the pixel selector data.
 */
inline void WritePixelData(__global uchar* block, uint pixel_data) {
	block[4] |= pixel_data >> 24;
	block[5] |= (pixel_data >> 16) & 0xff;
	block[6] |= (pixel_data >> 8) & 0xff;
	block[7] |= pixel_data & 0xff;
}

/**
 * @brief Sets the flip bit.
 */
inline void WriteFlip(__global uchar* block, bool flip) {
	block[3] &= ~0x01;
	block[3] |= (uchar)(flip);
}

/**
 * @brief Sets the differential bit.
 */
inline void WriteDiff(__global uchar* block, bool diff) {
	block[3] &= ~0x02;
	block[3] |= (uchar)(diff) << 1;
}

/**
 * @brief Creates a 444-bit color.
 */
inline Color makeColor444(const float* bgr) {
	uchar b4 = round_to_4_bits(bgr[0]);
	uchar g4 = round_to_4_bits(bgr[1]);
	uchar r4 = round_to_4_bits(bgr[2]);
	Color bgr444;
	bgr444.channels.b = (b4 << 4) | b4;
	bgr444.channels.g = (g4 << 4) | g4;
	bgr444.channels.r = (r4 << 4) | r4;
	bgr444.channels.a = 0x44;
	return bgr444;
}

/**
 * @brief Creates a 555-bit color.
 */
inline Color makeColor555(const float* bgr) {
	uchar b5 = round_to_5_bits(bgr[0]);
	uchar g5 = round_to_5_bits(bgr[1]);
	uchar r5 = round_to_5_bits(bgr[2]);
	Color bgr555;
	bgr555.channels.b = (b5 << 3) | (b5 >> 2);
	bgr555.channels.g = (g5 << 3) | (g5 >> 2);
	bgr555.channels.r = (r5 << 3) | (r5 >> 2);
	bgr555.channels.a = 0x55;
	return bgr555;
}
	
/**
 * @brief Calculates the average color of 8 pixels.
 */
void getAverageColor(const Color* src, float* avg_color)
{
	uint sum_b = 0, sum_g = 0, sum_r = 0;
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
 * @brief Optimizes luminance modifiers for a sub-block.
 */
unsigned long computeLuminance(__global uchar* block, const Color* src, const Color base, int sub_block_id, __constant uchar* idx_to_num_tab, unsigned long threshold)
{
	uint best_tbl_err = threshold;
	uchar best_tbl_idx = 0;
	uchar best_mod_idx[8][8];  

	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		Color candidate_color[4];
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			candidate_color[mod_idx] = makeColor(base, lum);
		}
		
		uint tbl_err = 0;
		for (unsigned int i = 0; i < 8; ++i) {
			uint best_mod_err = threshold;
			for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
				const Color color = candidate_color[mod_idx];
				uint mod_err = getColorError(src[i], color);
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

	uint pix_data = 0;
	for (unsigned int i = 0; i < 8; ++i) {
		uchar mod_idx = best_mod_idx[best_tbl_idx][i];
		uchar pix_idx = g_mod_to_pix[mod_idx];
		uint lsb = pix_idx & 0x1;
		uint msb = pix_idx >> 1;
		int texel_num = idx_to_num_tab[i];
		pix_data |= msb << (texel_num + 16);
		pix_data |= lsb << (texel_num);
	}
	WritePixelData(block, pix_data);
	return best_tbl_err;
}

/**
 * @brief Fast path for single-color blocks.
 */
bool tryCompressSolidBlock(__global uchar* dst, const Color* src, unsigned long* error)
{
	for (unsigned int i = 1; i < 16; ++i) {
		if (src[i].bits != src[0].bits) return false;
	}
	for (unsigned int i = 0; i < 8; i++) dst[i] = 0;

	float src_color_float[3] = {(float)(src->channels.b), (float)(src->channels.g), (float)(src->channels.r)};
	Color base = makeColor555(src_color_float);
	
	WriteDiff(dst, true);
	WriteFlip(dst, false);
	WriteColors555(dst, base, base);
	
	uchar best_tbl_idx = 0;
	uchar best_mod_idx = 0;
	uint best_mod_err = 0xFFFFFFFF; 
	
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			const Color color = makeColor(base, lum);
			uint mod_err = getColorError(*src, color);
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
	
	uchar pix_idx = g_mod_to_pix[best_mod_idx];
	uint lsb = pix_idx & 0x1;
	uint msb = pix_idx >> 1;
	
	uint pix_data = 0;
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
 * @brief Compresses a single 4x4 block.
 */
void compressBlock(__global uchar* dst, const Color* ver_src, const Color* hor_src, unsigned long threshold)
{
	unsigned long solid_error = 0;
	if (tryCompressSolidBlock(dst, ver_src, &solid_error)) return;
	
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
	
	uint sub_block_err[4] = {0};
	for (unsigned int i = 0; i < 4; ++i) {
		for (unsigned int j = 0; j < 8; ++j) {
			sub_block_err[i] += getColorError(sub_block_avg[i], sub_block_src[i][j]);
		}
	}
	
	bool flip = sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1];
	for (unsigned int i = 0; i < 8; i++) dst[i] = 0;
	
	WriteDiff(dst, use_differential[!!flip]);
	WriteFlip(dst, flip);
	
	uchar sub_block_off_0 = flip ? 2 : 0;
	uchar sub_block_off_1 = sub_block_off_0 + 1;
	
	if (use_differential[!!flip]) {
		WriteColors555(dst, sub_block_avg[sub_block_off_0], sub_block_avg[sub_block_off_1]);
	} else {
		WriteColors444(dst, sub_block_avg[sub_block_off_0], sub_block_avg[sub_block_off_1]);
	}
	
	computeLuminance(dst, sub_block_src[sub_block_off_0], sub_block_avg[sub_block_off_0], 0, g_idx_to_num[sub_block_off_0], threshold);
	computeLuminance(dst, sub_block_src[sub_block_off_1], sub_block_avg[sub_block_off_1], 1, g_idx_to_num[sub_block_off_1], threshold);
}

/**
 * @brief Kernel for parallel ETC1 texture compression.
 */
__kernel void compress(const __global uchar* src, __global uchar* dst, const int width, const int height)
{
	int x = get_global_id(0);
	int y = get_global_id(1);

	int depl_src = y * width * 16 + x * 16;
	int depl_dst = 8 * y * width / 4 + x * 8;

	Color ver_blocks[16], hor_blocks[16];
	Color row0[4], row1[4], row2[4], row3[4];

	for (int i = 0; i < 4; i++) {
		row0[i].channels.b = *(src + depl_src + 4 * i);
		row0[i].channels.g = *(src + depl_src + 4 * i + 1);
		row0[i].channels.r = *(src + depl_src + 4 * i + 2);
		row0[i].channels.a = *(src + depl_src + 4 * i + 3);
		
		row1[i].channels.b = *(src + depl_src + 4 * width + 4 * i);
		row1[i].channels.g = *(src + depl_src + 4 * width + 4 * i + 1);
		row1[i].channels.r = *(src + depl_src + 4 * width + 4 * i + 2);
		row1[i].channels.a = *(src + depl_src + 4 * width + 4 * i + 3);
		
		row2[i].channels.b = *(src + depl_src + 8 * width + 4 * i);
		row2[i].channels.g = *(src + depl_src + 8 * width + 4 * i + 1);
		row2[i].channels.r = *(src + depl_src + 8 * width + 4 * i + 2);
		row2[i].channels.a = *(src + depl_src + 8 * width + 4 * i + 3);
		
		row3[i].channels.b = *(src + depl_src + 12 * width + 4 * i);
		row3[i].channels.g = *(src + depl_src + 12 * width + 4 * i + 1);
		row3[i].channels.r = *(src + depl_src + 12 * width + 4 * i + 2);
		row3[i].channels.a = *(src + depl_src + 12 * width + 4 * i + 3);
	}

	for (int i = 0; i < 4; i++) {
		if (i < 2) {
			ver_blocks[i] = row0[i];
			ver_blocks[i + 2] = row1[i];
			ver_blocks[i + 4] = row2[i];
			ver_blocks[i + 6] = row3[i];
		} else {
			ver_blocks[i + 6] = row0[i];
			ver_blocks[i + 8] = row1[i];
			ver_blocks[i + 10] = row2[i];
			ver_blocks[i + 12] = row3[i];
		}
		hor_blocks[i] = row0[i];
		hor_blocks[i + 4] = row1[i];
		hor_blocks[i + 8] = row2[i];
		hor_blocks[i + 12] = row3[i];
	}

	compressBlock(dst + depl_dst, ver_blocks, hor_blocks, 0x7FFFFFFF);
}

>>>> file: texture_compress_skl.cpp
/**
 * @file texture_compress_skl.cpp
 * @brief Host-side TextureCompressor implementation.
 */

#include "compress.hpp"
#include "helper.hpp"
#include <iostream>
#include <vector>
#include <string>

using namespace std;

/**
 * @brief Initializes OpenCL and selects hardware.
 */
TextureCompressor::TextureCompressor() {	
	int selected = 0;
	cl_uint device_num = 0, platform_num = 0;
	cl_platform_id platform;

	CL_ERR( clGetPlatformIDs(0, NULL, &platform_num));
	platform_ids = new cl_platform_id[platform_num];
	CL_ERR( clGetPlatformIDs(platform_num, platform_ids, NULL));

	for (int platf = 0; platf < platform_num; platf++) {
		platform = platform_ids[platf];
		clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &device_num);
		if (device_num > 0) {
			device_ids = new cl_device_id[device_num];
			CL_ERR( clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, device_num, device_ids, NULL));
			device = device_ids[0];
			selected = 1;
			break;
		}
	}
}

/**
 * @brief Cleans up OpenCL resources.
 */
TextureCompressor::~TextureCompressor() {
	delete[] platform_ids;
	delete[] device_ids;
}
	
/**
 * @brief Compresses texture using GPU-accelerated ETC1.
 */
unsigned long TextureCompressor::compress(const uint8_t* src, uint8_t* dst, int width, int height) {
	cl_mem src_in, dst_out;
	int ret;
	string kernel_src;

	context = clCreateContext(0, 1, &device, NULL, NULL, &ret);
	command_queue = clCreateCommandQueue(context, device, 0, &ret);
	read_kernel("texture_compress_device.cl", kernel_src);
	const char* kernel_c_str = kernel_src.c_str();

	program = clCreateProgramWithSource(context, 1, &kernel_c_str, NULL, &ret);
	clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	kernel = clCreateKernel(program, "compress", &ret);
	
	src_in = clCreateBuffer(context, CL_MEM_READ_ONLY, width * height * 4, NULL, NULL);
	dst_out = clCreateBuffer(context, CL_MEM_READ_WRITE, width * height / 2, NULL, NULL);

	clEnqueueWriteBuffer(command_queue, src_in, CL_TRUE, 0, width * height * 4, src, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, dst_out, CL_TRUE, 0, width * height / 2, dst, 0, NULL, NULL);

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &src_in);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &dst_out);
	clSetKernelArg(kernel, 2, sizeof(int), &width);
	clSetKernelArg(kernel, 3, sizeof(int), &height);

	size_t globalSize[2] = {(size_t)(width / 4), (size_t)(height / 4)};
	clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, globalSize, 0, 0, NULL, NULL);
	clFinish(command_queue);

	clEnqueueReadBuffer(command_queue, dst_out, CL_TRUE, 0, width * height / 2, dst, 0, NULL, NULL);

	clReleaseMemObject(src_in);
	clReleaseMemObject(dst_out);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);

	return 0;
}
