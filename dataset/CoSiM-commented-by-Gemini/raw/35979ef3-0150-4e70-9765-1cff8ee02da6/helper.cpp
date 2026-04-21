/**
 * @file helper.cpp
 * @brief OpenCL host utility infrastructure.
 * Functional Utility: Centralizes error handling, logging, and kernel I/O 
 * to simplify the orchestration of heterogeneous compute tasks.
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
 * @param cl_ret status code from an OpenCL function.
 * @return 0 on success, 1 on failure.
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
 * @brief Retrieves program build info for specific hardware diagnostics.
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
 * @brief Interface declarations for OpenCL utilities.
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
 * @file kernel_maria.cl
 * @brief OpenCL kernel for parallel ETC1 texture compression.
 * Algorithm: Parallel block-based compression optimized for GPU execution.
 */

/*
typedef union Color {
	struct BgraColorType {
		unsigned char b;
		unsigned char g;
		unsigned char r;
		unsigned char a;
	} channels;
	unsigned char components[4];
	unsigned int bits;
} Color;


// clamp_renamed: restricted value utility.
inline unsigned char clamp_renamed(unsigned char val, unsigned char min, unsigned char max) {
	return (unsigned char)(val < min ? min : (val > max ? max : val));
}


inline int clamp_renamed_int(int val, int min, int max) {
	return (int)(val < min ? min : (val > max ? max : val));
}

// round_to_5_bits: Quantizes channel to 5-bit depth.
inline unsigned char round_to_5_bits(float val) {
	return clamp_renamed((unsigned char)(val * 31.0f / 255.0f + 0.5f), (unsigned char)0, (unsigned char)31);
}

// round_to_4_bits: Quantizes channel to 4-bit depth.
inline unsigned char round_to_4_bits(float val) {
	return clamp_renamed((unsigned char)(val * 15.0f / 255.0f + 0.5f), (unsigned char)0, (unsigned char)15);
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

__constant unsigned char g_mod_to_pix[4] = {3, 2, 0, 1};

// g_idx_to_num: maps block-local layout to specification indexing.
__constant unsigned char g_idx_to_num[4][8] = {
	{0, 4, 1, 5, 2, 6, 3, 7},        // Vertical block 0.
	{8, 12, 9, 13, 10, 14, 11, 15},  // Vertical block 1.
	{0, 4, 8, 12, 1, 5, 9, 13},      // Horizontal block 0.
	{2, 6, 10, 14, 3, 7, 11, 15}     // Horizontal block 1.
};

// makeColor: Derives final color point from base and luminance modifier.
inline Color makeColor(const Color base, short lum) {
	int b = (int)(base.channels.b) + lum;


	int g = (int)(base.channels.g) + lum;
	int r = (int)(base.channels.r) + lum;
	Color color;
	color.channels.b = (unsigned char)(clamp_renamed_int(b, 0, 255));
	color.channels.g = (unsigned char)(clamp_renamed_int(g, 0, 255));
	color.channels.r = (unsigned char)(clamp_renamed_int(r, 0, 255));
	return color;
}

// getColorError: Evaluates similarity using perceptual weighting or Euclidean distance.
inline unsigned int getColorError(const Color u, const Color v) {
#ifdef USE_PERCEIVED_ERROR_METRIC
	float delta_b = (float)(u.channels.b) - v.channels.b;


	float delta_g = (float)(u.channels.g) - v.channels.g;
	float delta_r = (float)(u.channels.r) - v.channels.r;
	return (unsigned int)(0.299f * delta_b * delta_b +
								 0.587f * delta_g * delta_g +
								 0.114f * delta_r * delta_r);
#else
	int delta_b = (int)(u.channels.b) - v.channels.b;
	int delta_g = (int)(u.channels.g) - v.channels.g;
	int delta_r = (int)(u.channels.r) - v.channels.r;
	return delta_b * delta_b + delta_g * delta_g + delta_r * delta_r;
#endif
}



inline void WriteColors444(global unsigned char* block,
						   const Color color0,
						   const Color color1) {
	block[0] = (color0.channels.r & 0xf0) | (color1.channels.r >> 4);
	block[1] = (color0.channels.g & 0xf0) | (color1.channels.g >> 4);
	block[2] = (color0.channels.b & 0xf0) | (color1.channels.b >> 4);
}

static inline void WriteColors555(global unsigned char* block,
						   const Color color0,


						   const Color color1) {
	unsigned char two_compl_trans_table[8] = {
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



inline void WriteCodewordTable(global unsigned char* block,
							   unsigned char sub_block_id,
							   unsigned char table) {
	
	unsigned char shift = (2 + (3 - sub_block_id * 3));
	block[3] &= ~(0x07 << shift);
	block[3] |= table << shift;
}



inline void WritePixelData(global unsigned char* block, unsigned int pixel_data) {
	block[4] |= pixel_data >> 24;
	block[5] |= (pixel_data >> 16) & 0xff;
	block[6] |= (pixel_data >> 8) & 0xff;
	block[7] |= pixel_data & 0xff;
}

inline void WriteFlip(global unsigned char* block, bool flip) {
	block[3] &= ~0x01;
	block[3] |= (unsigned char)(flip);
}

inline void WriteDiff(global unsigned char* block, bool diff) {
	block[3] &= ~0x02;
	block[3] |= (unsigned char)(diff) << 1;
}

// my_memcpy: Custom memcpy for kernel private memory space.
void my_memcpy(void *dst, const void *src, size_t size)
{
   char *pdst = dst;
   const char *psrc = src;
   for (size_t i=0; i<size; i++) 
      pdst[i] = psrc[i];
}

// ExtractBlock: populates 4x4 texel grid from image source.
inline void ExtractBlock(unsigned char* dst, const unsigned char* src, int width) {
	for (int j = 0; j < 4; ++j) {
		my_memcpy(&dst[j * 4 * 4], src, 4 * 4);


		src += width * 4;
	}
}

// makeColor444: individual color quantization.
inline Color makeColor444(const float* bgr) {
	unsigned char b4 = round_to_4_bits(bgr[0]);
	unsigned char g4 = round_to_4_bits(bgr[1]);
	unsigned char r4 = round_to_4_bits(bgr[2]);
	Color bgr444;
	bgr444.channels.b = (b4 << 4) | b4;
	bgr444.channels.g = (g4 << 4) | g4;


	bgr444.channels.r = (r4 << 4) | r4;
	bgr444.channels.a = 0x44;
	return bgr444;
}

// makeColor555: differential color quantization.
inline Color makeColor555(const float* bgr) {
	unsigned char b5 = round_to_5_bits(bgr[0]);
	unsigned char g5 = round_to_5_bits(bgr[1]);
	unsigned char r5 = round_to_5_bits(bgr[2]);
	Color bgr555;
	bgr555.channels.b = (b5 > 2);
	bgr555.channels.g = (g5 > 2);


	bgr555.channels.r = (r5 > 2);
	bgr555.channels.a = 0x55;
	return bgr555;
}
	
// getAverageColor: Estimates energy center of sub-block.
void getAverageColor(const Color* src, float* avg_color)
{
	unsigned int sum_b = 0, sum_g = 0, sum_r = 0;
	
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
	
// computeLuminance: Brute-force optimal codeword search.
unsigned long computeLuminance(global unsigned char* block,
						   const Color* src,
						   const Color base,
						   int sub_block_id,
						   __constant unsigned char* idx_to_num_tab,
						   unsigned long threshold)
{
	unsigned int best_tbl_err = threshold;
	unsigned char best_tbl_idx = 0;
	unsigned char best_mod_idx[8][8];  

	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		Color candidate_color[4];  


		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			candidate_color[mod_idx] = makeColor(base, lum);
		}
		
		unsigned int tbl_err = 0;
		
		for (unsigned int i = 0; i < 8; ++i) {
			unsigned int best_mod_err = threshold;


			for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
				const Color color = candidate_color[mod_idx];
				
				unsigned int mod_err = getColorError(src[i], color);
				if (mod_err < best_mod_err) {
					best_mod_idx[tbl_idx][i] = mod_idx;
					best_mod_err = mod_err;
					
					if (mod_err == 0)
						break;  
				}
			}
			
			tbl_err += best_mod_err;
			if (tbl_err > best_tbl_err)


				break;  
		}
		
		if (tbl_err < best_tbl_err) {
			best_tbl_err = tbl_err;
			best_tbl_idx = tbl_idx;
			
			if (tbl_err == 0)
				break;  
		}
	}

	WriteCodewordTable(block, sub_block_id, best_tbl_idx);

	unsigned int pix_data = 0;

	for (unsigned int i = 0; i < 8; ++i) {


		unsigned char mod_idx = best_mod_idx[best_tbl_idx][i];
		unsigned char pix_idx = g_mod_to_pix[mod_idx];
		
		unsigned int lsb = pix_idx & 0x1;
		unsigned int msb = pix_idx >> 1;
		
		int texel_num = idx_to_num_tab[i];
		pix_data |= msb << (texel_num + 16);
		pix_data |= lsb << (texel_num);
	}

	WritePixelData(block, pix_data);

	return best_tbl_err;
}


// tryCompressSolidBlock: Performance path for uniform blocks.
bool tryCompressSolidBlock(global unsigned char* dst,
						   const Color* src,
						   unsigned long* error)
{
	for (unsigned int i = 1; i < 16; ++i) {
		if (src[i].bits != src[0].bits)
			return false;
	}
	
   	for (size_t i=0; i<8; i++) 
    	dst[i] = (unsigned char)0;
	
	float src_color_float[3] = {(float)(src->channels.b),
		(float)(src->channels.g),


		(float)(src->channels.r)};
	Color base = makeColor555(src_color_float);
	
	WriteDiff(dst, true);
	WriteFlip(dst, false);
	WriteColors555(dst, base, base);
	
	unsigned char best_tbl_idx = 0;
	unsigned char best_mod_idx = 0;
	unsigned int best_mod_err = 4294967295; 
	
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];


			const Color color = makeColor(base, lum);
			
			unsigned int mod_err = getColorError(*src, color);
			if (mod_err < best_mod_err) {
				best_tbl_idx = tbl_idx;
				best_mod_idx = mod_idx;
				best_mod_err = mod_err;
				
				if (mod_err == 0)
					break;  
			}
		}
		
		if (best_mod_err == 0)
			break;
	}
	
	WriteCodewordTable(dst, 0, best_tbl_idx);
	WriteCodewordTable(dst, 1, best_tbl_idx);
	
	unsigned char pix_idx = g_mod_to_pix[best_mod_idx];
	unsigned int lsb = pix_idx & 0x1;
	unsigned int msb = pix_idx >> 1;
	
	unsigned int pix_data = 0;
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

// compressBlock: Full ETC1 logic for 4x4 block.
unsigned long compressBlock(global unsigned char* dst,
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
	
	unsigned int sub_block_err[4] = {0};
	for (unsigned int i = 0; i < 4; ++i) {
		for (unsigned int j = 0; j < 8; ++j) {
			sub_block_err[i] += getColorError(sub_block_avg[i], sub_block_src[i][j]);
		}
	}
	
	bool flip =
	sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1];
	
   	for (size_t i=0; i<8; i++) 
    	dst[i] = (unsigned char)0;
	


	WriteDiff(dst, use_differential[!!flip]);
	WriteFlip(dst, flip);
	
	unsigned char sub_block_off_0 = flip ? 2 : 0;
	unsigned char sub_block_off_1 = sub_block_off_0 + 1;
	
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
	
// compress: OpenCL kernel for parallel block compression.
__kernel void compress(__global uint8 *src,
						__global uint8 *dst,
						int width,
						int height) {
	

	int h = get_global_id(0);
	int w = get_global_id(1);

	Color ver_blocks[16];
	Color hor_blocks[16];
	
	unsigned long compressed_error = 0;
	
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			const Color* row0 = ((unsigned char)src + y * (width * 4 * 4) + x * 4 *4);
			const Color* row1 = row0 + width;
			const Color* row2 = row1 + width;
			const Color* row3 = row2 + width;
			
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
			
			compressed_error += compressBlock((unsigned char)(dst + x * 8), ver_blocks, hor_blocks, 4294967295);
		}
	}	
	return;
}
*/

/**
 * @file texture_compress_maria.cpp
 * @brief Host management logic for OpenCL compression.
 */

/*
#include "compress.hpp"

#include <iostream>
#include <vector>
#include <CL/cl.h>

#include "helper.hpp"

using namespace std;

// gpu_find: Automatically discovers and selects compute device.
void gpu_find(cl_device_id &device)
{
	cl_platform_id platform;
	cl_uint platform_num = 0;
	cl_uint device_num = 0;
	int platform_index;
	size_t attr_size = 0;
	cl_char* attr_data = NULL;

	clGetPlatformIDs(0, NULL, &platform_num);
	cl_platform_id* platform_list = new cl_platform_id[platform_num];
	clGetPlatformIDs(platform_num, platform_list, NULL);

	for(uint platf=0; platf<platform_num; platf++)
	{
		clGetPlatformInfo(platform_list[platf], CL_PLATFORM_VENDOR, 0, NULL, &attr_size);
		attr_data = new cl_char[attr_size];
		clGetPlatformInfo(platform_list[platf], CL_PLATFORM_VENDOR, attr_size, attr_data, NULL);
		
		if (strncmp("NVIDIA", (char *)attr_data, 6) == 0) {
			platform_index = platf;
		}

		delete[] attr_data;
		platform = platform_list[platf];

		if(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &device_num) == CL_DEVICE_NOT_FOUND) {
			device_num = 0;
			continue;
		}

		cl_device_id* device_list = new cl_device_id[device_num];
		clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, device_num, device_list, NULL);

		for(uint dev=0; dev<device_num; dev++)
		{
			if ((platf == platform_index) && (dev == 0)){
				device = device_list[dev];
			}
		}
	}
}


TextureCompressor::TextureCompressor() { 	
	cl_device_id device;
	gpu_find(device);
	this->device = device;
}

TextureCompressor::~TextureCompressor() { }
	
unsigned long TextureCompressor::compress(const uint8_t* src,
									  uint8_t* dst,
									  int width,
									  int height)
{
	cl_int ret;
	cl_context context;
	cl_command_queue cmd_queue;
	string kernel_src;
	cl_kernel kernel;
	cl_program program;
	int size = width * height * 4;

	context = clCreateContext(0, 1, &this->device, NULL, NULL, &ret);
	cmd_queue = clCreateCommandQueue(context, this->device, 0, &ret);

	cl_mem srcBuf = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(uint8_t) * size, NULL, &ret);
    cl_mem dstBuf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint8_t) * size / 8, NULL, &ret);
    
    clEnqueueWriteBuffer(cmd_queue, srcBuf, CL_TRUE, 0, sizeof(uint8_t) * size, src, 0, NULL, NULL);

	read_kernel("kernel_maria.cl", kernel_src);
	const char* kernel_c_str = kernel_src.c_str();

	program = clCreateProgramWithSource(context, 1, &kernel_c_str, NULL, &ret);
	clBuildProgram(program, 1, &this->device, "", NULL, NULL);

	kernel = clCreateKernel(program, "compress", &ret);

	clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&srcBuf);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&dstBuf);
	clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&width);
	clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&height);

	size_t globalSize[2] = {(size_t)height / 4, (size_t)width / 4};
	clEnqueueNDRangeKernel(cmd_queue, kernel, 2, NULL, globalSize, 0, 0, NULL, NULL);

	clEnqueueReadBuffer(cmd_queue, dstBuf, CL_TRUE, 0, sizeof(uint8_t) * size / 8, dst, 0, NULL, NULL);

	clFinish(cmd_queue);

	clReleaseMemObject(srcBuf);
	clReleaseMemObject(dstBuf);
	clReleaseCommandQueue(cmd_queue);
	clReleaseContext(context);
	return 0;
}
*/
