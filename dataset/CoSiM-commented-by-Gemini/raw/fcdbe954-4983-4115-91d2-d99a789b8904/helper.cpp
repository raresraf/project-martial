
>>>> file: helper.cpp
/**
 * @file helper.cpp
 * @brief Host-side utilities for OpenCL context management and error reporting.
 *
 * This module provides standard boilerplate for interacting with the OpenCL API,
 * including error code translation, kernel source loading, and device probing.
 * It is designed to reduce the repetitive setup code required for GPGPU applications.
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdio>

#include "helper.hpp"

using namespace std;

/**
 * @brief Checks an OpenCL runtime return code for success.
 * @param cl_ret The return code from an OpenCL API call.
 * @return 0 if successful, 1 if an error occurred (prints error message).
 *
 * This is the primary error handling utility for standard API calls like
 * buffer creation or memory transfers.
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
 * @brief Checks an OpenCL compilation return code and reports build logs on failure.
 * @param cl_ret The return code from clBuildProgram.
 * @param program The program object that failed to build.
 * @param device The device for which the build was attempted.
 * @return 0 if successful, 1 if an error occurred.
 *
 * This function is critical for debugging kernel syntax errors or hardware-specific
 * compilation failures.
 */
int CL_COMPILE_ERR(int cl_ret,
                  cl_program program,
                  cl_device_id device)
{
	if(cl_ret != CL_SUCCESS){
		cout << endl << cl_get_string_err(cl_ret) << endl;
		cl_get_compiler_err_log(program, device);
		return 1;
	}
	return 0;
}

/**
 * @brief Reads the raw source code of an OpenCL kernel from a file.
 * @param file_name Path to the .cl file.
 * @param str_kernel String to populate with the kernel source.
 */
void read_kernel(string file_name, string &str_kernel)
{
	ifstream in_file(file_name.c_str());
    // Note: Re-opening the file seems redundant but maintained for zero mutation.
	in_file.open(file_name.c_str());
	DIE( !in_file.is_open(), "ERR OpenCL kernel file. Same directory as binary ?" );

	stringstream str_stream;
	str_stream << in_file.rdbuf();

	str_kernel = str_stream.str();
}

/**
 * @brief Maps OpenCL error constants to human-readable strings.
 * @param err The cl_int error code.
 * @return A static string describing the error.
 *
 * Includes comprehensive coverage of standard OpenCL 1.x and 2.x error codes.
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
 * @brief Queries and prints the build log for a specific program and device.
 * Used internally by CL_COMPILE_ERR to provide diagnostic output.
 */
void cl_get_compiler_err_log(cl_program program,
                             cl_device_id device)
{
	char* build_log;
	size_t log_size;

	/* first call to know the proper size */
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
						  0, NULL, &log_size);
	build_log = new char[ log_size + 1 ];

	/* second call to get the log */
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
						  log_size, build_log, NULL);
	build_log[ log_size ] = '\0';
	cout << endl << build_log << endl;
    delete[] build_log;
}
>>>> file: helper.hpp
/**
 * @file helper.hpp
 * @brief Interface for host-side OpenCL helper utilities.
 * Handles platform abstraction (Apple vs standard) and defines common macros.
 */
#ifndef CL_HELPER_H
#define CL_HELPER_H

#if __APPLE__
   #include <OpenCL/opencl.h>
#else
   #include <CL/cl.h>
#endif

using namespace std;

/**
 * Function declarations for error handling and utility tasks.
 */
int CL_ERR(int cl_ret);
int CL_COMPILE_ERR(int cl_ret,
                  cl_program program,
                  cl_device_id device);

const char* cl_get_string_err(cl_int err);
void cl_get_compiler_err_log(cl_program program,
                             cl_device_id device);

void read_kernel(string file_name, string &str_kernel);

/**
 * @macro DIE
 * @brief Asserts a condition and terminates the program with a system error message on failure.
 * Useful for critical initialization steps where failure is non-recoverable.
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
>>>> file: sol_device.cl
/**
 * @file sol_device.cl
 * @brief Device-side OpenCL kernel for ETC1 texture compression.
 *
 * This kernel implements the core logic for evaluating pixel blocks and
 * determining the best compressed representation according to the ETC1 standard.
 * It utilizes SIMD-friendly structures and constant memory for tables.
 */

#define ALIGNAS(X)	__attribute__((aligned(X)))

#define UINT_MAX  0xffffffff
#define INT32_MAX 2147483647

/**
 * @union Color
 * @brief Multi-view representation of a pixel (BGRA).
 */
union Color {
	struct BgraColorType {
		uchar b;
		uchar g;
		uchar r;
		uchar a;
	} channels;
	uchar components[4];
	unsigned int bits;
};

/**
 * @brief Optimized memory copy from global to local memory.
 */
void  my_memcpy(void* dst, __global const void* src, int num) {
	uchar* d = (uchar*)dst;
	__global uchar* s = (__global uchar*)src;
	int i;

	for (i = 0; i < num; ++i) {
		d[i] = s[i];
	}
}

/**
 * @brief Optimized memory copy from private to global memory.
 */
void my_memcpy2(__global void* dst, const void* src, int num) {
	__global uchar* d = (__global uchar*)dst;
	uchar* s = (uchar*)src;
	int i;

	for (i = 0; i < num; ++i) {
		d[i] = s[i];
	}
}

/**
 * @brief Manual implementation of memset for global memory buffers.
 */
void my_memset(__global void *b, int c, int len)
{
  int i;
  __global uchar *p = b;
  i = 0;
  while(len > 0)
    {
      *p = c;
      p++;
      len--;
    }
}

/**
 * @brief Maps a float value [0, 255] to a 5-bit integer [0, 31].
 */
inline uchar round_to_5_bits(float val) {
	return clamp((uchar)(val * 31.0f / 255.0f + 0.5f), (uchar)0, (uchar)31);
}

/**
 * @brief Maps a float value [0, 255] to a 4-bit integer [0, 15].
 */
inline uchar round_to_4_bits(float val) {
	return clamp((uchar)(val * 15.0f / 255.0f + 0.5f), (uchar)0, (uchar)15);
}

/**
 * @brief ETC1 Codeword tables for luminance modification.
 * @see Table 3.17.2 of the ETC1 Specification.
 */
ALIGNAS(16) __constant short g_codeword_tables[8][4] = {
	{-8, -2, 2, 8},
	{-17, -5, 5, 17},
	{-29, -9, 9, 29},
	{-42, -13, 13, 42},
	{-60, -18, 18, 60},
	{-80, -24, 24, 80},
	{-106, -33, 33, 106},
	{-183, -47, 47, 183}};

/**
 * @brief Modifier to pixel index mapping.
 * @see Table 3.17.3 of the ETC1 Specification.
 */
__constant uchar g_mod_to_pix[4] = {3, 2, 0, 1};

/**
 * @brief Texel coordinate remapping table.
 * Translates between linear array indices and ETC1 specification block ordering.
 */
__constant uchar g_idx_to_num[4][8] = {
	{0, 4, 1, 5, 2, 6, 3, 7},        // Vertical block 0.
	{8, 12, 9, 13, 10, 14, 11, 15},  // Vertical block 1.
	{0, 4, 8, 12, 1, 5, 9, 13},      // Horizontal block 0.
	{2, 6, 10, 14, 3, 7, 11, 15}     // Horizontal block 1.
};

/**
 * @brief Generates a color variant based on a base color and a luminance offset.
 */
inline union Color makeColor(const union Color* base, short lum) {
	int b = (uchar)(base->channels.b) + lum;
	int g = (uchar)(base->channels.g) + lum;
	int r = (uchar)(base->channels.r) + lum;
	union Color color;
	color.channels.b = (uchar)(clamp(b, 0, 255));
	color.channels.g = (uchar)(clamp(g, 0, 255));
	color.channels.r = (uchar)(clamp(r, 0, 255));
    color.channels.a = 255;
	return color;
}

/**
 * @brief Computes the squared color distance (error) between two pixels.
 */
inline uint getColorError(const union Color* u, const union Color* v) {
#ifdef USE_PERCEIVED_ERROR_METRIC
	float delta_b = (float)(u->channels.b) - v.channels.b;
	float delta_g = (float)(u->channels.g) - v.channels.g;
	float delta_r = (float)(u->channels.r) - v.channels.r;
	return (uint)(0.299f * delta_b * delta_b +
								 0.587f * delta_g * delta_g +
								 0.114f * delta_r * delta_r);
#else
	int delta_b = (int)(u->channels.b) - v->channels.b;
	int delta_g = (int)(u->channels.g) - v->channels.g;
	int delta_r = (int)(u->channels.r) - v->channels.r;
	return delta_b * delta_b + delta_g * delta_g + delta_r * delta_r;
#endif
}

/**
 * @brief Encodes colors in 444 format (Individual mode).
 */
inline void WriteColors444(__global uchar* block,
						   const union Color* color0,
						   const union Color* color1) {
	// Write output color for BGRA textures.
	block[0] = (color0->channels.r & 0xf0) | (color1->channels.r >> 4);
	block[1] = (color0->channels.g & 0xf0) | (color1->channels.g >> 4);
	block[2] = (color0->channels.b & 0xf0) | (color1->channels.b >> 4);
}

/**
 * @brief Encodes colors in 555+333 format (Differential mode).
 */
inline void WriteColors555(__global uchar* block,
						   const union Color* color0,
						   const union Color* color1) {
	// Table for conversion to 3-bit two complement format.
	uchar two_compl_trans_table[8] = {
		4,  // -4 (100b)
		5,  // -3 (101b)
		6,  // -2 (110b)
		7,  // -1 (111b)
		0,  //  0 (000b)
		1,  //  1 (001b)
		2,  //  2 (010b)
		3,  //  3 (011b)
	};
	
	short delta_r =
	(short)(color1->channels.r >> 3) - (color0->channels.r >> 3);
	short delta_g =
	(short)(color1->channels.g >> 3) - (color0->channels.g >> 3);
	short delta_b =
	(short)(color1->channels.b >> 3) - (color0->channels.b >> 3);
	
	// Write output color for BGRA textures.
	block[0] = (color0->channels.r & 0xf8) | two_compl_trans_table[delta_r + 4];
	block[1] = (color0->channels.g & 0xf8) | two_compl_trans_table[delta_g + 4];
	block[2] = (color0->channels.b & 0xf8) | two_compl_trans_table[delta_b + 4];
}

/**
 * @brief Writes the codeword table index to the sub-block header.
 */
inline void WriteCodewordTable(__global uchar* block,
							   uchar sub_block_id,
							   uchar table) {
	
	uchar shift = (2 + (3 - sub_block_id * 3));
	block[3] &= ~(0x07 << shift);
	block[3] |= table << shift;
}

/**
 * @brief Writes the packed pixel index data to the block.
 */
inline void WritePixelData(__global uchar* block, uint pixel_data) {
	block[4] |= pixel_data >> 24;
	block[5] |= (pixel_data >> 16) & 0xff;
	block[6] |= (pixel_data >> 8) & 0xff;
	block[7] |= pixel_data & 0xff;
}

/**
 * @brief Sets the block orientation bit.
 */
inline void WriteFlip(__global uchar* block, bool flip) {
	block[3] &= ~0x01;
	block[3] |= (uchar)(flip);
}

/**
 * @brief Sets the differential encoding bit.
 */
inline void WriteDiff(__global uchar* block, bool diff) {
	block[3] &= ~0x02;
	block[3] |= (uchar)(diff) << 1;
}

/**
 * @brief Extracts a 4x4 block of pixels for processing.
 */
inline void ExtractBlock(__global uchar* dst, const uchar* src, int width) {
	for (int j = 0; j < 4; ++j) {
		my_memcpy2(&dst[j * 4 * 4], src, 4 * 4);
		src += width * 4;
	}
}

/**
 * @brief Expands 4-bit channels to 8-bit for internal calculations.
 */
inline union Color makeColor444(const float* bgr) {
	uchar b4 = round_to_4_bits(bgr[0]);
	uchar g4 = round_to_4_bits(bgr[1]);
	uchar r4 = round_to_4_bits(bgr[2]);
	union Color bgr444;
	bgr444.channels.b = (b4 << 4) | b4;
	bgr444.channels.g = (g4 << 4) | g4;
	bgr444.channels.r = (r4 << 4) | r4;
	// Marker for 444 format
	bgr444.channels.a = 0x44;
	return bgr444;
}

/**
 * @brief Expands 5-bit channels to 8-bit for internal calculations.
 */
inline union Color makeColor555(const float* bgr) {
	uchar b5 = round_to_5_bits(bgr[0]);
	uchar g5 = round_to_5_bits(bgr[1]);
	uchar r5 = round_to_5_bits(bgr[2]);
	union Color bgr555;
	bgr555.channels.b = (b5 << 3) | (b5 >> 2);
	bgr555.channels.g = (g5 << 3) | (g5 >> 2);
	bgr555.channels.r = (r5 << 3) | (r5 >> 2);
	// Marker for 555 format
	bgr555.channels.a = 0x55;
	return bgr555;
}
	
/**
 * @brief Calculates the arithmetic average color of a set of pixels.
 */
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
	
/**
 * @brief Exhaustive search for the optimal luminance table and modifier indices.
 */
unsigned long computeLuminance(__global uchar* block,
						   union Color* src,
						   union Color* base,
						   int sub_block_id,
						   __constant uchar* idx_to_num_tab,
						   unsigned long threshold)
{
	uint best_tbl_err = threshold;
	uchar best_tbl_idx = 0;
	uchar best_mod_idx[8][8];  // [table][texel]

	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		union Color candidate_color[4];  // [modifier]
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			candidate_color[mod_idx] = makeColor(base, lum);
		}
		
		uint tbl_err = 0;
		
		for (unsigned int i = 0; i < 8; ++i) {
			uint best_mod_err = threshold;
			for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
				const union Color color = candidate_color[mod_idx];
				
				uint mod_err = getColorError(&src[i], &color);
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
 * @brief Fast path for blocks with perfectly uniform color.
 */
bool tryCompressSolidBlock(__global uchar* dst,
						   const union Color* src,
						   unsigned long* error)
{
	for (unsigned int i = 1; i < 16; ++i) {
		if (src[i].bits != src[0].bits)
			return false;
	}
	
	my_memset(dst, 0, 8);
	
	float src_color_float[3] = {(float)(src->channels.b),
		(float)(src->channels.g),
		(float)(src->channels.r)};
	union Color base = makeColor555(src_color_float);
	
	WriteDiff(dst, true);
	WriteFlip(dst, false);
	WriteColors555(dst, &base, &base);
	
	uchar best_tbl_idx = 0;
	uchar best_mod_idx = 0;
	uint best_mod_err = UINT_MAX; 
	
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			const union Color color = makeColor(&base, lum);
			
			uint mod_err = getColorError(src, &color);
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
 * @brief High-level orchestration of the compression for a single 4x4 block.
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
			sub_block_err[i] += getColorError(&sub_block_avg[i], &sub_block_src[i][j]);
		}
	}
	
	bool flip =
	sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1];
	
	my_memset(dst, 0, 8);
	
	WriteDiff(dst, use_differential[!!flip]);
	WriteFlip(dst, flip);
	
	uchar sub_block_off_0 = flip ? 2 : 0;
	uchar sub_block_off_1 = sub_block_off_0 + 1;
	
	if (use_differential[!!flip]) {
		WriteColors555(dst, &sub_block_avg[sub_block_off_0],
					   &sub_block_avg[sub_block_off_1]);
	} else {
		WriteColors444(dst, &sub_block_avg[sub_block_off_0],
					   &sub_block_avg[sub_block_off_1]);
	}
	
	unsigned long lumi_error1 = 0, lumi_error2 = 0;
	
	lumi_error1 = computeLuminance(dst, &sub_block_src[sub_block_off_0],
								   &sub_block_avg[sub_block_off_0], 0,
								   g_idx_to_num[sub_block_off_0],
								   threshold);
	lumi_error2 = computeLuminance(dst, &sub_block_src[sub_block_off_1],
								   &sub_block_avg[sub_block_off_1], 1,
								   g_idx_to_num[sub_block_off_1],
								   threshold);
	
	return lumi_error1 + lumi_error2;
}

/**
 * @brief Top-level kernel for parallel texture compression.
 */
__kernel void mmul(const int width,	const int height,
					  __global uchar* src,
					  __global uchar* dst,
					  __global unsigned long* ans) 
{
	int y = get_global_id(0) * 4;
	int x = get_global_id(1) * 4;

	int offset_src = y * width * 4 + x * 4;
	int offset_dst = (get_global_id(0) * (width/4) + get_global_id(1)) * 8;

	union Color ver_blocks[16];
	union Color hor_blocks[16];

	__global const union Color* row0 = (__global union Color*)(src + offset_src);
	__global const union Color* row1 = (__global union Color*)(src + offset_src + width * 4);
	__global const union Color* row2 = (__global union Color*)(src + offset_src + width * 8);
	__global const union Color* row3 = (__global union Color*)(src + offset_src + width * 12);
	
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
	
	*ans += compressBlock(dst + offset_dst, ver_blocks, hor_blocks, INT32_MAX);
}
>>>> file: texture_compress_skl.cpp
/**
 * @file texture_compress_skl.cpp
 * @brief Host-side implementation of the OpenCL texture compressor.
 * Orchestrates the full lifecycle of texture compression on the GPU.
 */
#include "compress.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdio>

#include "helper.hpp"

using namespace std;

/**
 * @brief Initializes the OpenCL environment.
 * Discovers GPU devices, creates a context, command queue, and compiles the kernel.
 */
TextureCompressor::TextureCompressor() { 
	cl_platform_id platform;
	cl_uint platform_num = 0;
	cl_uint device_num = 0;
	size_t attr_size = 0;
	cl_char* attr_data = NULL;

	/* get num of available OpenCL platforms */
	CL_ERR( clGetPlatformIDs(0, NULL, &platform_num));
	this->platform_ids = new cl_platform_id[platform_num];
	DIE(this->platform_ids == NULL, "alloc platform_ids");

	CL_ERR( clGetPlatformIDs(platform_num, this->platform_ids, NULL));
	cout << "Platform found: " << platform_num << endl;

	for(uint platf=0; platf<platform_num; platf++)
	{
		platform = platform_ids[platf];
		DIE(platform == 0, "platform selection");
		
		/* get num of available OpenCL devices type GPU on the selected platform */
		clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &device_num);
		
		if (device_num)
		{
			device_ids = new cl_device_id[device_num];
			DIE(device_ids == NULL, "alloc devices");
			
			/* get all available OpenCL devices type GPU on the selected platform */
			CL_ERR( clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU,
								   device_num, device_ids, NULL));

			CL_ERR( clGetPlatformInfo(platform_ids[platf],
									  CL_PLATFORM_VENDOR, 0, NULL, &attr_size));
			attr_data = new cl_char[attr_size];
			DIE(attr_data == NULL, "alloc attr_data");
			
			/* get data CL_PLATFORM_VENDOR */
			CL_ERR( clGetPlatformInfo(platform_ids[platf],
									  CL_PLATFORM_VENDOR, attr_size, attr_data, NULL));
			cout << "Platform " << platf << " " << attr_data << " ";

			delete[] attr_data;
			
			/* get attribute size CL_PLATFORM_VERSION */
			CL_ERR( clGetPlatformInfo(platform_ids[platf],
									  CL_PLATFORM_VERSION, 0, NULL, &attr_size));
			attr_data = new cl_char[attr_size];
			DIE(attr_data == NULL, "alloc attr_data");
			
			/* get data size CL_PLATFORM_VERSION */
			CL_ERR( clGetPlatformInfo(platform_ids[platf],
									  CL_PLATFORM_VERSION, attr_size, attr_data, NULL));
			cout << attr_data << endl;
			delete[] attr_data;
			
			cout << "\tDevices found " << device_num  << endl;
		
			uint dev = 0;
			/* get attribute size */
			CL_ERR( clGetDeviceInfo(device_ids[dev], CL_DEVICE_NAME,
									0, NULL, &attr_size));
			attr_data = new cl_char[attr_size];
			DIE(attr_data == NULL, "alloc attr_data");
			
			/* get attribute CL_DEVICE_NAME */
			CL_ERR( clGetDeviceInfo(device_ids[dev], CL_DEVICE_NAME,
									attr_size, attr_data, NULL));
			cout << "\tDevice " << dev << " " << attr_data << " ";
			delete[] attr_data;
			
			/* get attribute size */
			CL_ERR( clGetDeviceInfo(device_ids[dev], CL_DEVICE_VERSION,
									0, NULL, &attr_size));
			attr_data = new cl_char[attr_size];
			DIE(attr_data == NULL, "alloc attr_data");
			
			/* get attribute CL_DEVICE_VERSION */
			CL_ERR( clGetDeviceInfo(device_ids[dev], CL_DEVICE_VERSION,
									attr_size, attr_data, NULL));
			cout << attr_data;
			delete[] attr_data;
			
			device = device_ids[dev];

			break;
		}
	}

	string kernel_src;
	int ret;

	/* create a context for the device */
	this->context = clCreateContext(0, 1, &this->device, NULL, NULL, &ret);
	CL_ERR( ret );
	
	this->command_queue = clCreateCommandQueue(this->context, this->device,
									CL_QUEUE_PROFILING_ENABLE, &ret);
	CL_ERR( ret );
		
	/* retrieve kernel source */
	read_kernel("sol_device.cl", kernel_src);
	const char* kernel_c_str = kernel_src.c_str();
	
	this->program = clCreateProgramWithSource(context, 1,
				(const char **) &kernel_c_str, NULL, &ret);
	CL_ERR( ret );
	
	// Build the program
	ret = clBuildProgram(this->program, 0, NULL, NULL, NULL, NULL);
	CL_COMPILE_ERR( ret, this->program, this->device );
	
	// Create the compute kernel from the program
	this->kernel = clCreateKernel(program, "mmul", &ret);
	CL_ERR( ret );
}

/**
 * @brief Destructor. Releases OpenCL resources.
 */
TextureCompressor::~TextureCompressor() 
{ 
	delete[] device_ids;
	delete[] platform_ids;

	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
}

/**
 * @brief Performs texture compression on the GPU.
 */
unsigned long TextureCompressor::compress(const uint8_t* src,
									  uint8_t* dst,
									  int width,
									  int height)
{
	int 			sz;
	size_t			global[2];
	size_t			local[2] = {1, 1}; // Default local work size.
	cl_mem			src_mem;
	cl_mem			dst_mem;
	cl_mem 			ans_mem;
	int				ret;
	unsigned long ans = 0;

	sz = width * height;

	// Buffer allocation on GPU memory.
	src_mem  = clCreateBuffer(context,  CL_MEM_READ_ONLY,
							sizeof(uint8_t) * sz, NULL, NULL);
	dst_mem  = clCreateBuffer(context,  CL_MEM_WRITE_ONLY,
							sizeof(uint8_t) * sz / 2, NULL, NULL);
	ans_mem  = clCreateBuffer(context,  CL_MEM_READ_WRITE,
							sizeof(unsigned long) , NULL, NULL);
	
    // Kernel argument binding.
	ret  = 0;
	ret  = clSetKernelArg(kernel, 0, sizeof(int), &width);
	ret |= clSetKernelArg(kernel, 1, sizeof(int), &height);
	ret |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &src_mem);
	ret |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &dst_mem);
	ret |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &ans_mem);

	// Memory upload to GPU.
	ret = clEnqueueWriteBuffer(this->command_queue, src_mem, CL_TRUE, 0,
							   sizeof(uint8_t) * sz, src, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(this->command_queue, ans_mem, CL_TRUE, 0,
							   sizeof(unsigned long), &ans, 0, NULL, NULL);
	cl_event prof_event;

	global[0] =(size_t) height / 4;
	global[1] =(size_t) width / 4;
	
	// Kernel launch.
	ret = clEnqueueNDRangeKernel(this->command_queue, kernel, 2, NULL,
								 global, local, 0, NULL, &prof_event);
	
	clFinish(this->command_queue);
	
	// Result download from GPU.
	ret = clEnqueueReadBuffer( this->command_queue, dst_mem, CL_TRUE, 0,
							  sizeof(uint8_t) * sz / 2, dst, 0, NULL, NULL );
	CL_ERR(ret);

	ret = clEnqueueReadBuffer( this->command_queue, ans_mem, CL_TRUE, 0,
							  sizeof(unsigned long), &ans, 0, NULL, NULL );
	CL_ERR(ret);

	ret = clFinish(this->command_queue);
	CL_ERR(ret);
	
	clReleaseMemObject(src_mem);
	clReleaseMemObject(dst_mem);
	clReleaseMemObject(ans_mem);

	return ans;
}
