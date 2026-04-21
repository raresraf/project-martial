/**
 * @file defines.hpp
 * @brief Configuration and implementation for OpenCL ETC1 texture compression.
 * 
 * Domain-Aware: Implements GPU-accelerated block compression for 4x4 texel grids.
 * Optimization: Uses OpenCL kernel to parallelize per-block error minimization.
 */

#ifndef DEFINES_H
#define DEFINES_H

#define SELECT_VENDOR "NVIDIA"
#define DEVICE_NO 1
#define KERNEL_FILE "kernel.cl"
#define KERNEL_FUNCTION "start"

#endif

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "helper.hpp"

using namespace std;

/**
 * @brief Helper to handle OpenCL errors by terminating execution with diagnostic info.
 */
void CL_ERR(int cl_ret)
{
  DIE(cl_ret != CL_SUCCESS, cl_get_string_err(cl_ret));
}

/**
 * @brief Checks for compilation errors in OpenCL kernels and prints logs.
 */
int CL_COMPILE_ERR(int cl_ret, cl_program program, cl_device_id device)
{
	/**
	 * Block Logic: Post-compilation state validation.
	 * Invariant: Triggers log retrieval only if build status is not CL_SUCCESS.
	 */
	if(cl_ret != CL_SUCCESS){
		cout << endl << cl_get_string_err(cl_ret) << endl;
		cl_get_compiler_err_log(program, device);
		return 1;
	}
	return 0;
}

/**
 * @brief Loads kernel source from a file.
 */
void read_kernel(string file_name, string &str_kernel)
{
	ifstream in_file(file_name.c_str());
	in_file.open(file_name.c_str());
	DIE( !in_file.is_open(), "ERR OpenCL kernel file. Same directory as binary ?" );

	stringstream str_stream;
	str_stream << in_file.rdbuf();
	str_kernel = str_stream.str();
}

/**
 * @brief Maps OpenCL numeric codes to human-readable error strings.
 */
const char* cl_get_string_err(cl_int err) {
switch (err) {
  case CL_SUCCESS:                     	return  "Success!";
...
  default:                                return  "Unknown";
  }
}

/**
 * @brief Retrieves the compiler's build log from the GPU driver.
 */
void cl_get_compiler_err_log(cl_program program, cl_device_id device)
{
	char* build_log;
	size_t log_size;
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
	build_log = new char[ log_size + 1 ];
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
	build_log[ log_size ] = '\0';
	cout << endl << build_log << endl;
}

#ifndef CL_HELPER_H
#define CL_HELPER_H

#if __APPLE__
   #include <OpenCL/opencl.h>
#else
   #include <CL/cl.h>
#endif

#include <cstdio>

using namespace std;

void CL_ERR(int cl_ret);
int CL_COMPILE_ERR(int cl_ret, cl_program program, cl_device_id device);
const char* cl_get_string_err(cl_int err);
void cl_get_compiler_err_log(cl_program program, cl_device_id device);
void read_kernel(string file_name, string &str_kernel);

#define DIE(assertion, call_description)  \
do { \
	if (assertion) { \
		fprintf(stderr, "(%d): ", __LINE__); \
		perror(call_description); \
		exit(EXIT_FAILURE); \
	} \
} while(0);

#endif

#define INT32_MAX 0xffffffff

/**
 * @brief BGRA color structure for ETC1 processing.
 */
typedef union Color {
	struct BgraColorType {
		uchar b; uchar g; uchar r; uchar a;
	} channels;
	uchar components[4];
	uint bits;
} Color;

/**
 * @brief ETC1 specification luminance tables.
 */
__constant short g_codeword_tables[8][4] = {
	{-8, -2, 2, 8}, {-17, -5, 5, 17}, {-29, -9, 9, 29}, {-42, -13, 13, 42},
	{-60, -18, 18, 60}, {-80, -24, 24, 80}, {-106, -33, 33, 106}, {-183, -47, 47, 183}};

__constant uchar g_mod_to_pix[4] = {3, 2, 0, 1};

__constant uchar g_idx_to_num[4][8] = {
	{0, 4, 1, 5, 2, 6, 3, 7}, {8, 12, 9, 13, 10, 14, 11, 15},  
	{0, 4, 8, 12, 1, 5, 9, 13}, {2, 6, 10, 14, 3, 7, 11, 15}     
};

void memset_global(__global uchar *dst, uchar value, size_t size) {
	size_t i;
	/**
	 * Block Logic: Memory fill.
	 */
	for (i = 0; i < size; ++i) {
		dst[i] = value;
	}
}

void memcpy_global_private(uchar *dst, __global uchar *src, size_t size) {
	size_t i;
	/**
	 * Block Logic: Global to Private memory migration.
	 */
	for (i = 0; i < size; ++i) {
		dst[i] = src[i];
	}
}

uchar clamp_int_to_uchar(int val, uchar min, uchar max) {
	return val < min ? min : (val > max ? max : val);
}

uchar round_to_5_bits(float val) {
	return clamp_int_to_uchar(val * 31.0f / 255.0f + 0.5f, 0, 31);
}

uchar round_to_4_bits(float val) {
	return clamp_int_to_uchar(val * 15.0f / 255.0f + 0.5f, 0, 15);
}

/**
 * @brief Entry point for parallel ETC1 block compression.
 * Memory Hierarchy: Reads from global 'src', uses private registers for block processing, writes to global 'dst'.
 */
__kernel void start(const __global uchar *src, __global uchar *dst, __global uint *total_err, const int width, const int height)
{
    int block_row = get_global_id(0);
    int block_col = get_global_id(1);
    int block_src = block_row * width * 16 + block_col * 16;
    int block_dst = block_row * get_global_size(1) * 8 + block_col * 8;

    Color ver_blocks[16];
	Color hor_blocks[16];

    const __global Color *row0 = (const __global Color*)(src + block_src);
    const __global Color *row1 = row0 + width;
    const __global Color *row2 = row1 + width;
    const __global Color *row3 = row2 + width;
    
	// Optimization: Loads sub-blocks into private memory to minimize global stalls during optimization search.
	memcpy_global_private((uchar*)ver_blocks, (__global uchar*)row0, 8);
	memcpy_global_private((uchar*)(ver_blocks + 2), (__global uchar*)row1, 8);
	memcpy_global_private((uchar*)(ver_blocks + 4), (__global uchar*)row2, 8);
	memcpy_global_private((uchar*)(ver_blocks + 6), (__global uchar*)row3, 8);
	memcpy_global_private((uchar*)(ver_blocks + 8), (__global uchar*)(row0 + 2), 8);
	memcpy_global_private((uchar*)(ver_blocks + 10), (__global uchar*)(row1 + 2), 8);
	memcpy_global_private((uchar*)(ver_blocks + 12), (__global uchar*)(row2 + 2), 8);
	memcpy_global_private((uchar*)(ver_blocks + 14), (__global uchar*)(row3 + 2), 8);
    
	memcpy_global_private((uchar*)hor_blocks, (__global uchar*)row0, 16);
	memcpy_global_private((uchar*)(hor_blocks + 4), (__global uchar*)row1, 16);
	memcpy_global_private((uchar*)(hor_blocks + 8), (__global uchar*)row2, 16);
	memcpy_global_private((uchar*)(hor_blocks + 12), (__global uchar*)row3, 16);

	memset_global(dst + block_dst, 0, 8);

	uint block_err = compressBlock(dst + block_dst, ver_blocks, hor_blocks, INT32_MAX);
	atomic_add(total_err, block_err); // Synchronization: Atomically updates global error accumulator.
}

/**
 * @brief Heuristically chooses between flip modes and encodings for a 4x4 block.
 */
uint compressBlock(__global uchar *dst, Color *ver_src, Color *hor_src, uint threshold) {
	Color* sub_block_src[4] = {ver_src, ver_src + 8, hor_src, hor_src + 8};
	Color sub_block_avg[4];
	uchar use_differential[2] = {1, 1};
    int i, j;

	/**
	 * Block Logic: Mode evaluation (Differential vs Standard).
	 * Invariant: Determines if component deltas fit in 3-bit differential fields.
	 */
	for (i = 0, j = 1; i < 4; i += 2, j += 2) {
		float avg_color_0[3], avg_color_1[3];
		getAverageColor(sub_block_src[i], avg_color_0);
		getAverageColor(sub_block_src[j], avg_color_1);
		Color avg_color_555_0 = makeColor555(avg_color_0);
		Color avg_color_555_1 = makeColor555(avg_color_1);

		for (uint light_idx = 0; light_idx < 3; ++light_idx) {
			int component_diff = (int)(avg_color_555_1.components[light_idx] >> 3) - (int)(avg_color_555_0.components[light_idx] >> 3);
			if (component_diff < -4 || component_diff > 3) {
				use_differential[i / 2] = 0;
				sub_block_avg[i] = makeColor444(avg_color_0);
				sub_block_avg[j] = makeColor444(avg_color_1);
			} else {
				sub_block_avg[i] = avg_color_555_0;
				sub_block_avg[j] = avg_color_555_1;
			}
		}
	}
	
	uint sub_block_err[4] = {0};
	for (i = 0; i < 4; ++i) {
		for (j = 0; j < 8; ++j) {
			sub_block_err[i] += getColorError(sub_block_avg[i], sub_block_src[i][j]);
		}
	}
	
    uchar flip = (sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1]) ? 1 : 0;
	WriteDiff(dst, use_differential[flip]);
	WriteFlip(dst, flip);
	
	uchar off0 = flip ? 2 : 0;
	uchar off1 = off0 + 1;
	
	if (use_differential[flip]) {
		WriteColors555(dst, sub_block_avg[off0], sub_block_avg[off1]);
	} else {
		WriteColors444(dst, sub_block_avg[off0], sub_block_avg[off1]);
	}
	
	uint err1 = computeLuminance(dst, sub_block_src[off0], sub_block_avg[off0], 0, g_idx_to_num[off0], threshold);
	uint err2 = computeLuminance(dst, sub_block_src[off1], sub_block_avg[off1], 1, g_idx_to_num[off1], threshold);
	return err1 + err2;
}

/**
 * @brief Optimizes luminance modifiers to minimize perceptual color distance.
 */
uint computeLuminance(__global uchar *block, Color *src, Color base, int sub_block_id, __constant uchar *idx_to_num_tab, uint threshold)
{
	uint best_tbl_err = threshold;
	uchar best_tbl_idx = 0;
	uchar best_mod_idx[8][8];  
    int i, tbl_idx, mod_idx;

	/**
	 * Block Logic: Exhaustive table search.
	 */
	for (tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		Color candidate_color[4];  
		for (mod_idx = 0; mod_idx < 4; ++mod_idx) {
			candidate_color[mod_idx] = makeColor(base, g_codeword_tables[tbl_idx][mod_idx]);
		}
		uint tbl_err = 0;
		for (i = 0; i < 8; ++i) {
			uint best_mod_err = threshold;
			for (mod_idx = 0; mod_idx < 4; ++mod_idx) {
				uint mod_err = getColorError(src[i], candidate_color[mod_idx]);
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
	for (i = 0; i < 8; ++i) {
		uchar mod = best_mod_idx[best_tbl_idx][i];
		uchar pix = g_mod_to_pix[mod];
		int texel = idx_to_num_tab[i];
		pix_data |= (uint)(pix & 0x1) << texel;
		pix_data |= (uint)(pix >> 1) << (texel + 16);
	}
	WritePixelData(block, pix_data);
	return best_tbl_err;
}

/**
 * @file lab8.cpp
 * @brief Implementation of OpenCL host routines for device discovery and grid orchestration.
 */

namespace lab8 {
/**
 * @brief Selects the optimal GPU device based on vendor and index.
 */
void gpu_device_find(cl_device_id &device, cl_platform_id* platform_list, cl_device_id* device_list, uint dev_no)
{
	cl_uint p_num = 0, d_num = 0;
	CL_ERR(clGetPlatformIDs(0, NULL, &p_num));
	platform_list = new cl_platform_id[p_num];
	CL_ERR(clGetPlatformIDs(p_num, platform_list, NULL));

	for(uint platf=0; platf<p_num; platf++)
	{
		size_t sz;
		clGetPlatformInfo(platform_list[platf], CL_PLATFORM_VENDOR, 0, NULL, &sz);
		char* vendor = new char[sz];
		clGetPlatformInfo(platform_list[platf], CL_PLATFORM_VENDOR, sz, vendor, NULL);
		bool match = (strstr(vendor, SELECT_VENDOR) != NULL);
		delete[] vendor;

		if (clGetDeviceIDs(platform_list[platf], CL_DEVICE_TYPE_ALL, 0, NULL, &d_num) == CL_SUCCESS) {
			device_list = new cl_device_id[d_num];
			clGetDeviceIDs(platform_list[platf], CL_DEVICE_TYPE_ALL, d_num, device_list, NULL);
			if (match && d_num > dev_no) {
				device = device_list[dev_no];
			}
		}
	}
}
}
