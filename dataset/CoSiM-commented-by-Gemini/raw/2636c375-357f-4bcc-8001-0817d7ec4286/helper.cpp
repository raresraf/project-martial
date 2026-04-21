/**
 * @file helper.cpp
 * @brief OpenCL Host infrastructure and error diagnostic tools.
 * Architectural Intent: Provides a reusable framework for managing OpenCL lifecycles, 
 * specialized for texture compression benchmarks.
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <CL/cl.h>

#include "helper.hpp"

using namespace std;

/**
 * @brief Validates OpenCL API return codes.
 */
int CL_ERR(int cl_ret)
{
    if(cl_ret != CL_SUCCESS){
        cout << endl << cl_get_string_err(cl_ret) << endl;
        return 1;
    }
    return 0;
}

/**
 * @brief Validates OpenCL compilation and reports build logs on failure.
 */
int CL_COMPILE_ERR(int cl_ret, cl_program program, cl_device_id device)
{
    if(cl_ret != CL_SUCCESS){
        cout << endl << cl_get_string_err(cl_ret) << endl;
        cl_get_compiler_err_log(program, device);
        return 1;
    }
    return 0;
}

/**
 * @brief Utility for loading kernel source strings from external storage.
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
 * @brief Mapping from OpenCL error codes to human-readable descriptors.
 */
const char* cl_get_string_err(cl_int err) {
    switch (err) {
        case CL_SUCCESS:                     	return  "Success!";
        case CL_DEVICE_NOT_FOUND:               return  "Device not found.";
        case CL_DEVICE_NOT_AVAILABLE:           return  "Device not available";
        case CL_COMPILER_NOT_AVAILABLE:         return  "Compiler not available";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:  return  "Memory object alloc fail";
        case CL_OUT_OF_RESOURCES:               return  "Out of resources";
        case CL_OUT_OF_HOST_MEMORY:             return  "Out of host memory";
        case CL_PROFILING_INFO_NOT_AVAILABLE:   return  "Profiling information N/A";
        case CL_MEM_COPY_OVERLAP:               return  "Memory copy overlap";
        case CL_IMAGE_FORMAT_MISMATCH:          return  "Image format mismatch";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:     return  "Image format no support";
        case CL_BUILD_PROGRAM_FAILURE:          return  "Program build failure";
        case CL_MAP_FAILURE:                    return  "Map failure";
        case CL_INVALID_VALUE:                  return  "Invalid value";
        case CL_INVALID_DEVICE_TYPE:            return  "Invalid device type";
        case CL_INVALID_PLATFORM:               return  "Invalid platform";
        case CL_INVALID_DEVICE:                 return  "Invalid device";
        case CL_INVALID_CONTEXT:                return  "Invalid context";
        case CL_INVALID_QUEUE_PROPERTIES:       return  "Invalid queue properties";
        case CL_INVALID_COMMAND_QUEUE:          return  "Invalid command queue";
        case CL_INVALID_HOST_PTR:               return  "Invalid host pointer";
        case CL_INVALID_MEM_OBJECT:             return  "Invalid memory object";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:return  "Invalid image format desc";
        case CL_INVALID_IMAGE_SIZE:             return  "Invalid image size";
        case CL_INVALID_SAMPLER:                return  "Invalid sampler";
        case CL_INVALID_BINARY:                 return  "Invalid binary";
        case CL_INVALID_BUILD_OPTIONS:          return  "Invalid build options";
        case CL_INVALID_PROGRAM:                return  "Invalid program";
        case CL_INVALID_PROGRAM_EXECUTABLE:     return  "Invalid program exec";
        case CL_INVALID_KERNEL_NAME:            return  "Invalid kernel name";
        case CL_INVALID_KERNEL_DEFINITION:      return  "Invalid kernel definition";
        case CL_INVALID_KERNEL:                 return  "Invalid kernel";
        case CL_INVALID_ARG_INDEX:              return  "Invalid argument index";
        case CL_INVALID_ARG_VALUE:              return  "Invalid argument value";
        case CL_INVALID_ARG_SIZE:               return  "Invalid argument size";
        case CL_INVALID_KERNEL_ARGS:            return  "Invalid kernel arguments";
        case CL_INVALID_WORK_DIMENSION:         return  "Invalid work dimension";
        case CL_INVALID_WORK_GROUP_SIZE:        return  "Invalid work group size";
        case CL_INVALID_WORK_ITEM_SIZE:         return  "Invalid work item size";
        case CL_INVALID_GLOBAL_OFFSET:          return  "Invalid global offset";
        case CL_INVALID_EVENT_WAIT_LIST:        return  "Invalid event wait list";
        case CL_INVALID_EVENT:                  return  "Invalid event";
        case CL_INVALID_OPERATION:              return  "Invalid operation";
        case CL_INVALID_GL_OBJECT:              return  "Invalid OpenGL object";
        case CL_INVALID_BUFFER_SIZE:            return  "Invalid buffer size";
        case CL_INVALID_MIP_LEVEL:              return  "Invalid mip-map level";
        default:                                return  "Unknown";
    }
}

/**
 * @brief Retrieves the build log for an OpenCL program.
 */
void cl_get_compiler_err_log(cl_program program, cl_device_id device)
{
    char* build_log;
    size_t log_size;

    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                          0, NULL, &log_size);
    build_log = new char[ log_size + 1 ];

    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                          log_size, build_log, NULL);
    build_log[ log_size ] = '\0';
    cout << endl << build_log << endl;
}

/**
 * @file helper.hpp
 * @brief Common declarations and error-abort macros.
 */

/*
#ifndef CL_HELPER_H
#define CL_HELPER_H

#if __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

using namespace std;

int CL_ERR(int cl_ret);
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
*/

/**
 * @file kernel_madalin.cl
 * @brief Kernel implementation for Ericsson Texture Compression (ETC1).
 */

/*
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

// clamputz: Range clamping for color components.
float clamputz(float val, float min, float max) {
	return val < min ? min : (val > max ? max : val);
}

// round_to_5_bits: Quantizes 8-bit channel to 5-bit space.
inline uchar round_to_5_bits(float val) {
	return (uchar)clamputz(val * 31.0f / 255.0f + 0.5f, 0, 31);
}

// round_to_4_bits: Quantizes 8-bit channel to 4-bit space.
inline uchar round_to_4_bits(float val) {
	return (uchar)clamputz(val * 15.0f / 255.0f + 0.5f, 0, 15);
}

// Codeword tables as defined in ETC1 spec.
__constant short g_codeword_tables[8][4] = {
	{-8, -2, 2, 8},
	{-17, -5, 5, 17},
	{-29, -9, 9, 29},
	{-42, -13, 13, 42},
	{-60, -18, 18, 60},
	{-80, -24, 24, 80},
	{-106, -33, 33, 106},
	{-183, -47, 47, 183}};

__constant uchar g_mod_to_pix[4] = {3, 2, 0, 1};

// g_idx_to_num: Translates local sub-block offsets to global 4x4 block indices.
__constant uchar g_idx_to_num[4][8] = {
	{0, 4, 1, 5, 2, 6, 3, 7},        // Vertical block 0.
	{8, 12, 9, 13, 10, 14, 11, 15},  // Vertical block 1.
	{0, 4, 8, 12, 1, 5, 9, 13},      // Horizontal block 0.
	{2, 6, 10, 14, 3, 7, 11, 15}     // Horizontal block 1.
};

// makeColor: Derives a specific luminance point from a base color.
inline union Color makeColor(__const union Color base, short lum) {
	int b = (int)(base.channels.b) + lum;
	int g = (int)(base.channels.g) + lum;
	int r = (int)(base.channels.r) + lum;
	union Color color;
	color.channels.b = (uchar)(clamp(b, 0, 255));
	color.channels.g = (uchar)(clamp(g, 0, 255));
	color.channels.r = (uchar)(clamp(r, 0, 255));
	return color;
}

// getColorError: Evaluates perceptual distance between two colors.
inline uint getColorError(__const union Color u, __const union Color v) {
#ifdef USE_PERCEIVED_ERROR_METRIC
	float delta_b = (float)(u.channels.b) - v.channels.b;
	float delta_g = (float)(u.channels.g) - v.channels.g;
	float delta_r = (float)(u.channels.r) - v.channels.r;
	return (uint)(0.299f * delta_b * delta_b +
								 0.587f * delta_g * delta_g +
								 0.114f * delta_r * delta_r);
#else
	int delta_b = (int)(u.channels.b) - v.channels.b;
	int delta_g = (int)(u.channels.g) - v.channels.g;
	int delta_r = (int)(u.channels.r) - v.channels.r;
	return delta_b * delta_b + delta_g * delta_g + delta_r * delta_r;
#endif
}

// WriteColors444: Packs Individual mode color pairs.
inline void WriteColors444(uchar* block,
						   __const union Color color0,
						   __const union Color color1) {
	block[0] = (color0.channels.r & 0xf0) | (color1.channels.r >> 4);
	block[1] = (color0.channels.g & 0xf0) | (color1.channels.g >> 4);
	block[2] = (color0.channels.b & 0xf0) | (color1.channels.b >> 4);
}

// WriteColors555: Packs Differential mode color pairs using two's complement deltas.
inline void WriteColors555(uchar* block,
						   __const union Color color0,
						   __const union Color color1) {
	__const uchar two_compl_trans_table[8] = {
		4, 5, 6, 7, 0, 1, 2, 3,
	};
	
	short delta_r =
	(short)(color1.channels.r >> 3) - (color0.channels.r >> 3);
	short delta_g =
	(short)(color1.channels.g >> 3) - (color0.channels.g >> 3);
	short delta_b =
	(short)(color1.channels.b >> 3) - (color0.channels.b >> 3);
	
	block[0] = (color0.channels.r & 0xf8) | two_compl_trans_table[delta_r + 4];
	block[1] = (color0.channels.g & 0xf8) | two_compl_trans_table[delta_g + 4];
	block[2] = (color0.channels.b & 0xf8) | two_compl_trans_table[delta_b + 4];
}

// WriteCodewordTable: Encodes table selection for a sub-block.
inline void WriteCodewordTable(uchar* block,
							   uchar sub_block_id,
							   uchar table) {
	
	uchar shift = (2 + (3 - sub_block_id * 3));
	block[3] &= ~(0x07 << shift);
	block[3] |= table << shift;
}

// WritePixelData: Packs 2-bit selectors into the block bitstream.
inline void WritePixelData(uchar* block, uint pixel_data) {
	block[4] |= pixel_data >> 24;
	block[5] |= (pixel_data >> 16) & 0xff;
	block[6] |= (pixel_data >> 8) & 0xff;
	block[7] |= pixel_data & 0xff;
}

// WriteFlip: Encodes sub-block orientation.
inline void WriteFlip(uchar* block, bool flip) {
	block[3] &= ~0x01;
	block[3] |= (uchar)(flip);
}

// WriteDiff: Encodes color mode selection.
inline void WriteDiff(uchar* block, bool diff) {
	block[3] &= ~0x02;
	block[3] |= (uchar)(diff) << 1;
}

// ExtractBlock: Maps linear buffer to 4x4 texel grid.
inline void ExtractBlock(uchar* dst, const uchar* src, int width) {
	for (int j = 0; j < 4; ++j) {
		for (int i = 0; i < 4 * 4; i++) {
            dst[j * 4 * 4 + i] = src[i];
		}
		src += width * 4;
	}
}

// makeColor444: individual mode quantization.
inline union Color makeColor444(const float* bgr) {
	uchar b4 = round_to_4_bits(bgr[0]);
	uchar g4 = round_to_4_bits(bgr[1]);
	uchar r4 = round_to_4_bits(bgr[2]);
	union Color bgr444;
	bgr444.channels.b = (b4 << 4) | b4;
	bgr444.channels.g = (g4 << 4) | g4;
	bgr444.channels.r = (r4 << 4) | r4;
	bgr444.channels.a = 0x44;
	return bgr444;
}

// makeColor555: differential mode quantization.
inline union Color makeColor555(const float* bgr) {
	uchar b5 = round_to_5_bits(bgr[0]);
	uchar g5 = round_to_5_bits(bgr[1]);
	uchar r5 = round_to_5_bits(bgr[2]);
	union Color bgr555;
	bgr555.channels.b = (b5 > 2);
	bgr555.channels.g = (g5 > 2);
	bgr555.channels.r = (r5 > 2);
	bgr555.channels.a = 0x55;
	return bgr555;
}

// getAverageColor: Estimates sub-block energy center.
void getAverageColor(const union Color* src, float* avg_color)
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

// computeLuminance: Brute-force optimization over codeword tables.
unsigned long computeLuminance(uchar* block,
						   __const union Color* src,
						   __const union Color base,
						   int sub_block_id,
						   __constant unsigned char* idx_to_num_tab,
						   unsigned long threshold)
{
	uint best_tbl_err = threshold;
	uchar best_tbl_idx = 0;
	uchar best_mod_idx[8][8];  

	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		union Color candidate_color[4];  
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			candidate_color[mod_idx] = makeColor(base, lum);
		}
		
		uint tbl_err = 0;
		for (unsigned int i = 0; i < 8; ++i) {
			uint best_mod_err = threshold;
			for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
				__const union Color color = candidate_color[mod_idx];
				
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

// tryCompressSolidBlock: Optimization for homogeneous regions.
bool tryCompressSolidBlock(uchar* dst,
						   __const union Color* src,
						   unsigned long* error)
{
	for (unsigned int i = 1; i < 16; ++i) {
		if (src[i].bits != src[0].bits)
			return false;
	}
	
	for (int i = 0 ; i < 8; i++) {
	    dst[i] = 0;
	}
	
	float src_color_float[3] = {(float)(src->channels.b),
		(float)(src->channels.g),
		(float)(src->channels.r)};
	union Color base = makeColor555(src_color_float);
	
	WriteDiff(dst, true);
	WriteFlip(dst, false);
	WriteColors555(dst, base, base);
	
	uchar best_tbl_idx = 0;
	uchar best_mod_idx = 0;
	uint best_mod_err = 4294967295;
	
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			__const union Color color = makeColor(base, lum);
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

// compressBlock: Full structural logic for ETC1 block encoding.
unsigned long compressBlock(__global uchar* dst,
                           __const union Color* ver_src,
                           __const union Color* hor_src,
                           unsigned long threshold)
{
	unsigned long solid_error = 0;
	if (tryCompressSolidBlock(dst, ver_src, &solid_error)) {
		return solid_error;
	}
	
	__const union  Color* sub_block_src[4] = {ver_src, ver_src + 8, hor_src, hor_src + 8};
	union Color sub_block_avg[4];
	bool use_differential[2] = {true, true};
	
	for (unsigned int i = 0, j = 1; i < 4; i += 2, j += 2) {
		float avg_color_0[3];
		getAverageColor(sub_block_src[i], avg_color_0);
		union Color avg_color_555_0 = makeColor555(avg_color_0);
		float avg_color_1[3];
		getAverageColor(sub_block_src[j], avg_color_1);
		union Color avg_color_555_1 = makeColor555(avg_color_1);
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
	
	for (int z = 0; z < 8; z++) {
	    dst[z] = 0;
	}
	WriteDiff(dst, use_differential[!!flip]);
	WriteFlip(dst, flip);
	uchar sub_block_off_0 = flip ? 2 : 0;
	uchar sub_block_off_1 = sub_block_off_0 + 1;
	if (use_differential[!!flip]) {
		WriteColors555(dst, sub_block_avg[sub_block_off_0],
					   sub_block_avg[sub_block_off_1]);
	} else {
		WriteColors444(dst, sub_block_avg[sub_block_off_0],
					   sub_block_avg[sub_block_off_1]);
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

void my_memcpy(union Color* dst, const union Color* src, int num) {
    for (int i = 0 ; i < num; i++) {
        dst[i] = src[i];
    }
}

// kernel_compress_block: Massively parallel work-item processing for texture compression.
__kernel void kernel_compress_block( __global uchar* src,
                                    __global uchar* dst,
                                    __global uint* width,
                                    __global uint* height,
                                    __global float* buf_error )
{
	uint gid = get_global_id(0);
    union Color ver_blocks[16];
    union Color hor_blocks[16];

    const union Color* row0 = src + 4 * gid / width[0];
    const union Color* row1 = row0 + width[0];
    const union Color* row2 = row1 + width[0];
    const union Color* row3 = row2 + width[0];

    my_memcpy(ver_blocks, row0, 8);
    my_memcpy(ver_blocks + 2, row1, 8);
    my_memcpy(ver_blocks + 4, row2, 8);
    my_memcpy(ver_blocks + 6, row3, 8);
    my_memcpy(ver_blocks + 8, row0 + 2, 8);
    my_memcpy(ver_blocks + 10, row1 + 2, 8);
    my_memcpy(ver_blocks + 12, row2 + 2, 8);
    my_memcpy(ver_blocks + 14, row3 + 2, 8);

    my_memcpy(hor_blocks, row0, 16);
    my_memcpy(hor_blocks + 4, row1, 16);
    my_memcpy(hor_blocks + 8, row2, 16);
    my_memcpy(hor_blocks + 12, row3, 16);

    atomic_add(buf_error[0], compressBlock(dst, ver_blocks, hor_blocks, 4294967295));
}
*/

/**
 * @file texture_compress_skl.cpp
 * @brief Host-side orchestration for OpenCL texture compression tasks.
 */

/*
#include "compress.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <CL/cl.h>

#include "helper.hpp"

using namespace std;

// gpu_find: Automatically detects and selects a suitable GPU for kernel execution.
void TextureCompressor::init_gpu() {
    cl_int ret;
    cl_uint platform_num = 0;
    cl_uint device_num = 0;
    size_t attr_size = 0;
    cl_char* attr_data = NULL;

    clGetPlatformIDs(0, NULL, &platform_num);
    platform_ids = new cl_platform_id[platform_num];
    clGetPlatformIDs(platform_num, platform_ids, NULL);

    for(uint platf = 0; platf < platform_num; platf++)
    {
        clGetPlatformInfo(platform_ids[platf], CL_PLATFORM_VENDOR, 0, NULL, &attr_size);
        attr_data = new cl_char[attr_size];
        clGetPlatformInfo(platform_ids[platf], CL_PLATFORM_VENDOR, attr_size, attr_data, NULL);
        delete[] attr_data;

        if(clGetDeviceIDs(platform_ids[platf], CL_DEVICE_TYPE_GPU, 0, NULL, &device_num) == CL_DEVICE_NOT_FOUND) {
            device_num = 0;
            continue;
        }

        device_ids = new cl_device_id[device_num];
        clGetDeviceIDs(platform_ids[platf], CL_DEVICE_TYPE_GPU, device_num, device_ids, NULL);

        if (device_num > 0) {
            this->device = device_ids[0];
            break;
        }
    }
}

TextureCompressor::TextureCompressor() {
    init_gpu();
}

TextureCompressor::~TextureCompressor() { }
	
unsigned long TextureCompressor::compress(const uint8_t* src,
									  uint8_t* dst,
									  int width,
									  int height)
{
    unsigned long compressed_error = 0;
    cl_int ret;
    string kernel_src;
    int gSize = width * height / 16;

    this->context = clCreateContext(0, 1, &(this->device), NULL, NULL, &ret);
    this->command_queue = clCreateCommandQueue(this->context, this->device, CL_QUEUE_PROFILING_ENABLE, &ret);

    cl_mem dev_src = clCreateBuffer(this->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uint8_t) * width * height * 4, (void*)src, &ret);
    cl_mem dev_dst = clCreateBuffer(this->context, CL_MEM_WRITE_ONLY, sizeof(uint8_t) * width * height * 4 / 8, NULL, &ret);
    
    read_kernel("kernel.cl", kernel_src);
    const char* kernel_c_str = kernel_src.c_str();

    this->program = clCreateProgramWithSource(this->context, 1, &kernel_c_str, NULL, &ret);
    ret = clBuildProgram(this->program, 1, &(this->device), "", NULL, NULL);
    CL_COMPILE_ERR( ret, this->program, this->device );

    this->kernel = clCreateKernel(program, "kernel_compress_block", &ret);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&dev_src);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&dev_dst);
    clSetKernelArg(kernel, 2, sizeof(int), (void *)&width);
    clSetKernelArg(kernel, 3, sizeof(int), (void *)&height);

    size_t globalSize = gSize;
    clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);

    clFinish(command_queue);
    clEnqueueReadBuffer(command_queue, dev_dst, CL_TRUE, 0, sizeof(uint8_t) * width * height * 4 / 8, dst, 0, NULL, NULL);

    clReleaseMemObject(dev_src);
    clReleaseMemObject(dev_dst);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    return 0;
}
*/
