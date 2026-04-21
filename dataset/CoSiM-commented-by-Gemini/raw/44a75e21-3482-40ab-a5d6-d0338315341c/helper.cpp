/**
 * @file helper.cpp
 * @brief OpenCL host utility infrastructure.
 * Functional Utility: Provides standardized error diagnostics, kernel management, 
 * and hardware discovery services to support heterogeneous compute orchestration.
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
 * @brief Handles build failures by retrieving and displaying compiler logs.
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
 * @brief Loads OpenCL kernel source from an external text file.
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
 * @brief Translates OpenCL error codes to human-readable strings.
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
 * @brief Retrieves build log for specific hardware.
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
 * @brief Header for OpenCL utility constants and declarations.
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
 * @file texture_compress_skl.cl
 * @brief OpenCL kernel logic for parallel ETC1 texture compression.
 * Architectural Intent: Implements the Ericsson Texture Compression (ETC1) standard
 * optimized for GPU SIMD execution using parallel work-items for 4x4 texel blocks.
 */

/*
#define ALIGNAS(X)	__attribute__((aligned(X)))
#define UINT32_MAX 0xffffffff
#define INT32_MAX 2147483647

// @struct Color
// @brief BGRA pixel representation with bitfield overlays.
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

// puneZero: Buffer clearing utility for global memory.
void puneZero(__global uchar* dst, int count)
{
	int i;
	for(i = 0; i < count; i++)
		dst[i] = 0;
}

// copiaza: Data movement utility for private staging.
void copiaza(union Color* destinatie, __global union Color* sursa, int count)
{
	int i;
	for(i = 0; i < count; i++)
		destinatie[i] = sursa[i];
}

// clamp1: restricted range utility.
uchar clamp1(uchar val, uchar min, uchar max) {
	return (uchar)(val < min ? min : (val > max ? max : val));
}

// clamp2: integer range utility.
int clamp2(int val, int min, int max)
{
	return (int)(val < min ? min : (val > max ? max : val));	
}

// round_to_5_bits: Quantizes channel to 5-bit depth.
uchar round_to_5_bits(float val) {
	return (uchar)clamp1(val * 31.0f / 255.0f + 0.5f, 0, 31);
}

// round_to_4_bits: Quantizes channel to 4-bit depth.
uchar round_to_4_bits(float val) {
	return (uchar)clamp1(val * 15.0f / 255.0f + 0.5f, 0, 15);
}

// Standard ETC1 luminance modifier tables.
ALIGNAS(16) __constant short g_codeword_tables[8][4] = {
	{-8, -2, 2, 8},
	{-17, -5, 5, 17},
	{-29, -9, 9, 29},
	{-42, -13, 13, 42},
	{-60, -18, 18, 60},
	{-80, -24, 24, 80},
	{-106, -33, 33, 106},
	{-183, -47, 47, 183}};

__constant uchar g_mod_to_pix[4] = {3, 2, 0, 1};

// g_idx_to_num: maps 4x4 texel offsets to ETC1 internal numbering.
__constant uchar g_idx_to_num[4][8] = {
	{0, 4, 1, 5, 2, 6, 3, 7},        // Vertical block 0.
	{8, 12, 9, 13, 10, 14, 11, 15},  // Vertical block 1.


	{0, 4, 8, 12, 1, 5, 9, 13},      // Horizontal block 0.
	{2, 6, 10, 14, 3, 7, 11, 15}     // Horizontal block 1.
};

// makeColor: Derives candidate colors from base and modifier.
 union Color makeColor(const union Color base, short lum) {
	int b = (int)(base.channels.b) + lum;
	int g = (int)(base.channels.g) + lum;
	int r = (int)(base.channels.r) + lum;
	union Color color;
	color.channels.b = (uchar)(clamp2(b, 0, 255));


	color.channels.g = (uchar)(clamp2(g, 0, 255));
	color.channels.r = (uchar)(clamp2(r, 0, 255));
	return color;
}

// getColorError: Evaluates perceptual distance between color points.
 int getColorError(const union Color u, const union Color v) {
#ifdef USE_PERCEIVED_ERROR_METRIC
	float delta_b = (float)(u.channels.b) - v.channels.b;
	float delta_g = (float)(u.channels.g) - v.channels.g;
	float delta_r = (float)(u.channels.r) - v.channels.r;
	return (int)(0.299f * delta_b * delta_b +
								 0.587f * delta_g * delta_g +
								 0.114f * delta_r * delta_r);
#else
	int delta_b = (int)(u.channels.b) - v.channels.b;
	int delta_g = (int)(u.channels.g) - v.channels.g;


	int delta_r = (int)(u.channels.r) - v.channels.r;
	return delta_b * delta_b + delta_g * delta_g + delta_r * delta_r;
#endif
}

 void WriteColors444(__global uchar* block,
						   union Color color0,
						   union Color color1) {
	block[0] = (color0.channels.r & 0xf0) | (color1.channels.r >> 4);
	block[1] = (color0.channels.g & 0xf0) | (color1.channels.g >> 4);


	block[2] = (color0.channels.b & 0xf0) | (color1.channels.b >> 4);
}

void WriteColors555(__global uchar* block,
						   union Color color0,
						   union Color color1) {
	uchar two_compl_trans_table[8] = {
		4, 5, 6, 7, 0, 1, 2, 3,
	};
	
	short delta_r = (short)(color1.channels.r >> 3) - (color0.channels.r >> 3);
	short delta_g = (short)(color1.channels.g >> 3) - (color0.channels.g >> 3);
	short delta_b = (short)(color1.channels.b >> 3) - (color0.channels.b >> 3);
	
	block[0] = (color0.channels.r & 0xf8) | two_compl_trans_table[delta_r + 4];
	block[1] = (color0.channels.g & 0xf8) | two_compl_trans_table[delta_g + 4];


	block[2] = (color0.channels.b & 0xf8) | two_compl_trans_table[delta_b + 4];
}

 void WriteCodewordTable(__global uchar* block,
							   uchar sub_block_id,
							   uchar table) {
	
	uchar shift = (2 + (3 - sub_block_id * 3));
	block[3] &= ~(0x07 << shift);
	block[3] |= table << shift;
}

 void WritePixelData(__global uchar* block, int pixel_data) {
	block[4] |= pixel_data >> 24;
	block[5] |= (pixel_data >> 16) & 0xff;
	block[6] |= (pixel_data >> 8) & 0xff;
	block[7] |= pixel_data & 0xff;
}

 void WriteFlip(__global uchar* block, bool flip) {
	block[3] &= ~0x01;
	block[3] |= (uchar)(flip);
}

 void WriteDiff(__global uchar* block, bool diff) {
	block[3] &= ~0x02;
	block[3] |= (uchar)(diff) << 1;
}

// makeColor444: Individual mode quantization.
 union Color makeColor444(const float* bgr) {
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

// makeColor555: Differential mode quantization.
 union Color makeColor555(const float* bgr) {
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
	
// getAverageColor: Energy center estimation for sub-blocks.
void getAverageColor(const union Color* src, float* avg_color)
{
	int sum_b = 0, sum_g = 0, sum_r = 0;
	
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
	
// computeLuminance: Brute-force optimal codeword search per block.
unsigned long computeLuminance(__global uchar* block,
						   const union Color* src,
						   const union Color base,
						   int sub_block_id,
						   __constant uchar* idx_to_num_tab,
						   unsigned long threshold)
{
	int best_tbl_err = threshold;
	uchar best_tbl_idx = 0;
	uchar best_mod_idx[8][8];  

	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		union Color candidate_color[4];  
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			candidate_color[mod_idx] = makeColor(base, lum);
		}
		
		int tbl_err = 0;
		for (unsigned int i = 0; i < 8; ++i) {
			int best_mod_err = threshold;
			for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
				const union Color color = candidate_color[mod_idx];
				int mod_err = getColorError(src[i], color);
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

	int pix_data = 0;
	for (unsigned int i = 0; i < 8; ++i) {
		uchar mod_idx = best_mod_idx[best_tbl_idx][i];
		uchar pix_idx = g_mod_to_pix[mod_idx];
		int lsb = pix_idx & 0x1;
		int msb = pix_idx >> 1;
		int texel_num = idx_to_num_tab[i];
		pix_data |= msb << (texel_num + 16);
		pix_data |= lsb << (texel_num);
	}
	WritePixelData(block, pix_data);
	return best_tbl_err;
}

/**
 * @brief Fast path for solid color blocks.
 */
bool tryCompressSolidBlock(__global uchar* dst,
						   const union Color* src,
						   unsigned long* error)
{
	for (unsigned int i = 1; i < 16; ++i) {
		if (src[i].bits != src[0].bits)
			return 0;
	}
	puneZero(dst, 8);
	float src_color_float[3] = {(float)(src->channels.b), (float)(src->channels.g), (float)(src->channels.r)};
	union Color base = makeColor555(src_color_float);
	WriteDiff(dst, 1);
	WriteFlip(dst, 0);
	WriteColors555(dst, base, base);
	uchar best_tbl_idx = 0;
	uchar best_mod_idx = 0;
	int best_mod_err = UINT32_MAX; 
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			const union Color color = makeColor(base, lum);
			int mod_err = getColorError(*src, color);
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
	int lsb = pix_idx & 0x1;
	int msb = pix_idx >> 1;
	int pix_data = 0;
	for (unsigned int i = 0; i < 2; ++i) {
		for (unsigned int j = 0; j < 8; ++j) {
			int texel_num = g_idx_to_num[i][j];
			pix_data |= msb << (texel_num + 16);
			pix_data |= lsb << (texel_num);
		}
	}
	WritePixelData(dst, pix_data);
	*error = 16 * best_mod_err;
	return 1;
}

/**
 * @brief Main structural encoding logic for ETC1 blocks.
 */
unsigned long compressBlock(__global uchar* dst,
							const union Color* ver_src,
							const union Color* hor_src,
							unsigned long threshold)
{
	unsigned long solid_error = 0;
	if (tryCompressSolidBlock(dst, ver_src, &solid_error)) {
		return solid_error;
	}
	const union Color* sub_block_src[4] = {ver_src, ver_src + 8, hor_src, hor_src + 8};
	union Color sub_block_avg[4];
	bool use_differential[2] = {1, 1};
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
				use_differential[i / 2] = 0;
				sub_block_avg[i] = makeColor444(avg_color_0);
				sub_block_avg[j] = makeColor444(avg_color_1);
			} else {
				sub_block_avg[i] = avg_color_555_0;
				sub_block_avg[j] = avg_color_555_1;
			}
		}
	}
	int sub_block_err[4] = {0};
	for (unsigned int i = 0; i < 4; ++i) {
		for (unsigned int j = 0; j < 8; ++j) {
			sub_block_err[i] += getColorError(sub_block_avg[i], sub_block_src[i][j]);
		}
	}
	bool flip = sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1];
	puneZero(dst, 8);
	WriteDiff(dst, use_differential[!!flip]);
	WriteFlip(dst, flip);
	uchar sub_block_off_0 = flip ? 2 : 0;
	uchar sub_block_off_1 = sub_block_off_0 + 1;
	if (use_differential[!!flip]) {
		WriteColors555(dst, sub_block_avg[sub_block_off_0], sub_block_avg[sub_block_off_1]);
	} else {
		WriteColors444(dst, sub_block_avg[sub_block_off_0], sub_block_avg[sub_block_off_1]);
	}
	unsigned long lumi_error1 = computeLuminance(dst, sub_block_src[sub_block_off_0], sub_block_avg[sub_block_off_0], 0, g_idx_to_num[sub_block_off_0], threshold);
	unsigned long lumi_error2 = computeLuminance(dst, sub_block_src[sub_block_off_1], sub_block_avg[sub_block_off_1], 1, g_idx_to_num[sub_block_off_1], threshold);
	return lumi_error1 + lumi_error2;
}

// compress: OpenCL kernel for massively parallel block encoding.
__kernel void compress(__global uchar* src,
                __global uchar* dst,
                const int width,
                const int height)
{
	const int y = get_global_id(0) *4;
    const int x = get_global_id(1) *4;
	union Color ver_blocks[16];
	union Color hor_blocks[16];
	__global union Color* row0;
	row0  = (__global union Color*)(src + y * width * 4 + x * 4);
	__global union Color* row1 = row0 + width;
	__global union Color* row2 = row1 + width;
	__global union Color* row3 = row2 + width;
	copiaza(ver_blocks, row0, 2);
	copiaza(ver_blocks + 2, row1, 2);
	copiaza(ver_blocks + 4, row2, 2);
	copiaza(ver_blocks + 6, row3, 2);
	copiaza(ver_blocks + 8, row0 + 2, 2);
	copiaza(ver_blocks + 10, row1 + 2, 2);
	copiaza(ver_blocks + 12, row2 + 2, 2);
	copiaza(ver_blocks + 14, row3 + 2, 2);
	copiaza(hor_blocks, row0, 4);
	copiaza(hor_blocks + 4, row1, 4);
	copiaza(hor_blocks + 8, row2, 4);
	copiaza(hor_blocks + 12, row3, 4);
	compressBlock(dst + (y * width / 4 + x) * 2, ver_blocks, hor_blocks, UINT32_MAX);
}
*/

/**
 * @file texture_compress_skl.cpp
 * @brief Host-side orchestration for OpenCL texture compression tasks.
 */

/*
#include "compress.hpp"
#include "helper.hpp"
#include <iostream>
#include <vector>
#include <CL/cl.h>

using namespace std;

// gpu_find: Automatically discovers and selects the appropriate compute device.
void gpu_find(cl_device_id &device)
{
	cl_int ret;
	cl_uint platform_num = 0;
	cl_platform_id* platform_list = NULL;
	cl_uint device_num = 0;
	cl_device_id* device_list = NULL;
	size_t attr_size = 0;
	cl_char* attr_data = NULL;
	ret = clGetPlatformIDs(0, NULL, &platform_num);
	platform_list = new cl_platform_id[platform_num];
	ret = clGetPlatformIDs(platform_num, platform_list, NULL);
	for(uint platf=0; platf<platform_num; platf++)
	{
		ret = clGetPlatformInfo(platform_list[platf],CL_PLATFORM_VENDOR, 0, NULL, &attr_size);
		attr_data = new cl_char[attr_size];
		ret = clGetPlatformInfo(platform_list[platf],CL_PLATFORM_VENDOR, attr_size, attr_data, NULL);
		delete[] attr_data;
		cl_platform_id platform = platform_list[platf];
		ret = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &device_num);
		device_list = new cl_device_id[device_num];
		ret = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL,device_num, device_list, NULL);
		for(uint dev=0; dev<device_num; dev++)
		{
			ret = clGetDeviceInfo(device_list[dev], CL_DEVICE_NAME, 0, NULL, &attr_size);
			attr_data = new cl_char[attr_size];
			ret = clGetDeviceInfo(device_list[dev], CL_DEVICE_NAME,attr_size, attr_data, NULL);
			if(strstr((char*) attr_data, "Tesla") != NULL)
				device = device_list[dev];
			delete[] attr_data;
		}
	}
}

TextureCompressor::TextureCompressor() { gpu_find(device); }
TextureCompressor::~TextureCompressor() { }

unsigned long TextureCompressor::compress(const uint8_t* src,
									  uint8_t* dst,
									  int width,
									  int height)
{
	size_t global[2];
	string kernel_src;
	cl_context context;
	cl_command_queue commands;
	cl_program program;
	cl_kernel kernel;
	cl_mem src_in, dst_out;
	int ret;
	context = clCreateContext(0, 1, &device, NULL, NULL, &ret);
	commands = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &ret);
	src_in = clCreateBuffer(context,  CL_MEM_READ_ONLY, sizeof(uint8_t) * width * height * 4, NULL, NULL);
	dst_out = clCreateBuffer(context,  CL_MEM_WRITE_ONLY, sizeof(uint8_t) * width * height * 4 / 8, NULL, NULL);
	read_kernel("texture_compress_skl.cl", kernel_src);
	const char* kernel_c_str = kernel_src.c_str();
	program = clCreateProgramWithSource(context, 1, (const char **) &kernel_c_str, NULL, &ret);
	ret = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	CL_COMPILE_ERR( ret, program, device );
	kernel = clCreateKernel(program, "compress", &ret);
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &src_in);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &dst_out);
	clSetKernelArg(kernel, 2, sizeof(int), &width);
	clSetKernelArg(kernel, 3, sizeof(int), &height);
	clEnqueueWriteBuffer(commands, src_in, CL_TRUE, 0, sizeof(uint8_t) * width * height * 4, src, 0, NULL, NULL);
	global[0] =(size_t) height / 4;
	global[1] =(size_t) width / 4;
	clEnqueueNDRangeKernel(commands, kernel, 2, NULL, global, 0, 0, NULL, NULL);
	clFinish(commands);
	clEnqueueReadBuffer( commands, dst_out, CL_TRUE, 0, sizeof(uint8_t) * width * height * 4 / 8, dst, 0, NULL, NULL );
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseMemObject(src_in);
	clReleaseMemObject(dst_out);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);
	return 0;
}
*/
