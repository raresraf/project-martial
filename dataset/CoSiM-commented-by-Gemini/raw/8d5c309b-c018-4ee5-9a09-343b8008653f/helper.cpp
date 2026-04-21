/**
 * @file helper.cpp
 * @brief OpenCL utility functions and ETC1 texture compression implementation.
 * 
 * This module provides a set of helper functions for OpenCL environment management,
 * error handling, and kernel loading. It also includes a comprehensive implementation
 * of the ETC1 (Ericsson Texture Compression) algorithm, designed to be executed
 * on both host and device (via OpenCL kernels).
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
 * Intercepts OpenCL return codes and prints a human-readable error message
 * if the operation was not successful.
 * 
 * @param cl_ret The OpenCL return code to check.
 * @return int Returns 1 if an error occurred, 0 otherwise.
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
 * @brief Checks and reports OpenCL program compilation errors.
 * 
 * In addition to reporting the error code, this function retrieves and prints
 * the build log for the specified program and device, which is essential for
 * debugging OpenCL kernel compilation issues.
 * 
 * @param cl_ret The OpenCL return code from the build operation.
 * @param program The OpenCL program object.
 * @param device The OpenCL device for which the program was built.
 * @return int Returns 1 if an error occurred, 0 otherwise.
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
 * @brief Reads an OpenCL kernel source from a file.
 * 
 * Opens a file containing OpenCL kernel source code and reads its entire
 * content into a string for later compilation.
 * 
 * @param file_name Path to the kernel source file.
 * @param str_kernel String reference to store the loaded kernel source.
 */
void read_kernel(string file_name, string &str_kernel)
{
	ifstream in_file(file_name.c_str());
	// Ensure the file is open; DIE macro handles fatal errors.
	DIE( !in_file.is_open(), "ERR OpenCL kernel file. Same directory as binary ?" );

	stringstream str_stream;
	str_stream << in_file.rdbuf();

	str_kernel = str_stream.str();
}

/**
 * @brief Translates OpenCL error codes into human-readable strings.
 * 
 * Provides a comprehensive mapping of OpenCL's integer error codes to their
 * corresponding symbolic names for easier diagnostics.
 * 
 * @param err The OpenCL error code.
 * @return const char* A string representation of the error.
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
 * @brief Retrieves and prints the compilation log for an OpenCL program.
 * 
 * Queries OpenCL for the build log associated with a specific program and device,
 * allocates the necessary buffer, and prints the log to stdout.
 * 
 * @param program The OpenCL program object.
 * @param device The OpenCL device.
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

// Note: The original file contained multiple sections separated by '>>>> file:' markers.
// For the purpose of this documentation task, I am maintaining the structure as found.

>>>> file: helper.hpp
#ifndef CL_HELPER_H
#define CL_HELPER_H

#if __APPLE__
   #include <OpenCL/opencl.h>
#else
   #include <CL/cl.h>
#endif

using namespace std;

/**
 * @brief OpenCL error checking utility.
 */
int CL_ERR(int cl_ret);

/**
 * @brief OpenCL compilation error checking utility.
 */
int CL_COMPILE_ERR(int cl_ret,
                  cl_program program,
                  cl_device_id device);

/**
 * @brief Error code to string translation.
 */
const char* cl_get_string_err(cl_int err);

/**
 * @brief Retrieval of OpenCL build logs.
 */
void cl_get_compiler_err_log(cl_program program,
                             cl_device_id device);

/**
 * @brief Utility to load kernel source from disk.
 */
void read_kernel(string file_name, string &str_kernel);

/**
 * @brief Macro for fatal error reporting and process termination.
 * 
 * If the assertion is true, it prints the line number and error description,
 * then exits the program.
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
 * @brief OpenCL kernel and device-side functions for ETC1 compression.
 * 
 * This file contains the core logic for ETC1 texture compression, including
 * color space conversions, luminance calculations, and block encoding.
 */

#define ALIGNAS(X)	__attribute__((aligned(X)))

#define UINT_MAX  0xffffffff
#define INT32_MAX 2147483647

/**
 * @brief Represents a 32-bit BGRA color.
 * 
 * Allows access to individual color channels, a component array, or the raw bit representation.
 */
union Color {
	struct BgraColorType {
		uchar b; /**< Blue channel */
		uchar g; /**< Green channel */
		uchar r; /**< Red channel */
		uchar a; /**< Alpha channel */
	} channels;
	uchar components[4]; /**< Array access to channels */
	unsigned int bits;   /**< Raw bitwise access */
};

/**
 * @brief Device-side memory copy from global memory to private/local memory.
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
 * @brief Device-side memory copy from private/local memory to global memory.
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
 * @brief Device-side memory set for global memory buffers.
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
 * @brief Quantizes a float color component to 5 bits.
 */
inline uchar round_to_5_bits(float val) {
	return clamp((uchar)(val * 31.0f / 255.0f + 0.5f), (uchar)0, (uchar)31);
}

/**
 * @brief Quantizes a float color component to 4 bits.
 */
inline uchar round_to_4_bits(float val) {
	return clamp((uchar)(val * 15.0f / 255.0f + 0.5f), (uchar)0, (uchar)15);
}

/**
 * @brief ETC1 Codeword tables for luminance adjustment.
 * 
 * Defined by the ETC1 specification (Table 3.17.2).
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
 * @brief Maps modifier indices to pixel index values.
 * 
 * Defined by the ETC1 specification (Table 3.17.3).
 */
__constant uchar g_mod_to_pix[4] = {3, 2, 0, 1};

/**
 * @brief Index mapping for sub-block extraction.
 * 
 * Translates natural array indices in a 4x4 block to the indices used by the ETC1 specification.
 */
__constant uchar g_idx_to_num[4][8] = {
	{0, 4, 1, 5, 2, 6, 3, 7},        // Vertical block 0.
	{8, 12, 9, 13, 10, 14, 11, 15},  // Vertical block 1.
	{0, 4, 8, 12, 1, 5, 9, 13},      // Horizontal block 0.
	{2, 6, 10, 14, 3, 7, 11, 15}     // Horizontal block 1.
};

/**
 * @brief Adjusts a base color with a luminance value.
 * 
 * Applies a luminance modifier to the R, G, and B channels of a color.
 * 
 * @param base Pointer to the base color.
 * @param lum The luminance adjustment value.
 * @return union Color The resulting adjusted color.
 */
inline union Color makeColor(const union Color* base, short lum) {
	int b = (uchar)(base->channels.b) + lum;
	int g = (uchar)(base->channels.g) + lum;
	int r = (uchar)(base->channels.r) + lum;
	union Color color;
	color.channels.b = (uchar)(clamp(b, 0, 255));
	color.channels.g = (uchar)(clamp(g, 0, 255));
	color.channels.r = (uchar)(clamp(r, 0, 255));
	return color;
}

/**
 * @brief Computes the error metric between two colors.
 * 
 * Uses either a perceived error metric (weighted channels) or a simple Euclidean distance.
 * 
 * @param u First color.
 * @param v Second color.
 * @return uint The calculated error.
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
 * @brief Writes encoded 444-format colors to the compressed block.
 */
inline void WriteColors444(__global uchar* block,
						   const union Color* color0,
						   const union Color* color1) {
	block[0] = (color0->channels.r & 0xf0) | (color1->channels.r >> 4);
	block[1] = (color0->channels.g & 0xf0) | (color1->channels.g >> 4);
	block[2] = (color0->channels.b & 0xf0) | (color1->channels.b >> 4);
}

/**
 * @brief Writes encoded 555-format colors with differential coding to the compressed block.
 */
inline void WriteColors555(__global uchar* block,
						   const union Color* color0,
						   const union Color* color1) {
	// Table for conversion to 3-bit two's complement format for the differential color.
	uchar two_compl_trans_table[8] = {4, 5, 6, 7, 0, 1, 2, 3};
	
	short delta_r = (short)(color1->channels.r >> 3) - (color0->channels.r >> 3);
	short delta_g = (short)(color1->channels.g >> 3) - (color0->channels.g >> 3);
	short delta_b = (short)(color1->channels.b >> 3) - (color0->channels.b >> 3);
	
	block[0] = (color0->channels.r & 0xf8) | two_compl_trans_table[delta_r + 4];
	block[1] = (color0->channels.g & 0xf8) | two_compl_trans_table[delta_g + 4];
	block[2] = (color0->channels.b & 0xf8) | two_compl_trans_table[delta_b + 4];
}

/**
 * @brief Writes the codeword table index for a sub-block into the encoded block.
 */
inline void WriteCodewordTable(__global uchar* block,
							   uchar sub_block_id,
							   uchar table) {
	uchar shift = (2 + (3 - sub_block_id * 3));
	block[3] &= ~(0x07 << shift);
	block[3] |= table << shift;
}

/**
 * @brief Writes the 32-bit pixel data (selectors) into the encoded block.
 */
inline void WritePixelData(__global uchar* block, uint pixel_data) {
	block[4] |= pixel_data >> 24;
	block[5] |= (pixel_data >> 16) & 0xff;
	block[6] |= (pixel_data >> 8) & 0xff;
	block[7] |= pixel_data & 0xff;
}

/**
 * @brief Sets the 'flip' bit in the encoded block (horizontal vs vertical split).
 */
inline void WriteFlip(__global uchar* block, bool flip) {
	block[3] &= ~0x01;
	block[3] |= (uchar)(flip);
}

/**
 * @brief Sets the 'diff' bit in the encoded block (differential vs individual mode).
 */
inline void WriteDiff(__global uchar* block, bool diff) {
	block[3] &= ~0x02;
	block[3] |= (uchar)(diff) << 1;
}

/**
 * @brief Extracts a 4x4 pixel block from a source texture.
 */
inline void ExtractBlock(__global uchar* dst, const uchar* src, int width) {
	for (int j = 0; j < 4; ++j) {
		my_memcpy2(&dst[j * 4 * 4], src, 4 * 4);
		src += width * 4;
	}
}

/**
 * @brief Creates a 444-format color and expands it back to 888.
 */
inline union Color makeColor444(const float* bgr) {
	uchar b4 = round_to_4_bits(bgr[0]);
	uchar g4 = round_to_4_bits(bgr[1]);
	uchar r4 = round_to_4_bits(bgr[2]);
	union Color bgr444;
	bgr444.channels.b = (b4 << 4) | b4;
	bgr444.channels.g = (g4 << 4) | g4;
	bgr444.channels.r = (r4 << 4) | r4;
	bgr444.channels.a = 0x44; // Marker for 444
	return bgr444;
}

/**
 * @brief Creates a 555-format color and expands it back to 888.
 */
inline union Color makeColor555(const float* bgr) {
	uchar b5 = round_to_5_bits(bgr[0]);
	uchar g5 = round_to_5_bits(bgr[1]);
	uchar r5 = round_to_5_bits(bgr[2]);
	union Color bgr555;
	bgr555.channels.b = (b5 << 3) | (b5 >> 2);
	bgr555.channels.g = (g5 << 3) | (g5 >> 2);
	bgr555.channels.r = (r5 << 3) | (r5 >> 2);
	bgr555.channels.a = 0x55; // Marker for 555
	return bgr555;
}
	
/**
 * @brief Calculates the average color of a sub-block.
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
 * @brief Exhaustively searches for the best luminance codeword table and pixel modifiers.
 * 
 * Iterates through all possible codeword tables and modifiers to minimize the error
 * for a specific sub-block.
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
		union Color candidate_color[4];
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
 * @brief Optimizes compression for blocks containing only a single uniform color.
 */
bool tryCompressSolidBlock(__global uchar* dst,
						   const union Color* src,
						   unsigned long* error)
{
	for (unsigned int i = 1; i < 16; ++i) {
		if (src[i].bits != src[0].bits) return false;
	}
	
	my_memset(dst, 0, 8);
	float src_color_float[3] = {(float)(src->channels.b), (float)(src->channels.g), (float)(src->channels.r)};
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
 * @brief Core ETC1 block compression logic.
 * 
 * Decides between horizontal and vertical splits, and differential vs individual coding.
 */
unsigned long compressBlock(__global uchar* dst,
						   const union Color* ver_src,
						   const union Color* hor_src,
						   unsigned long threshold)
{
	unsigned long solid_error = 0;
	if (tryCompressSolidBlock(dst, ver_src, &solid_error)) return solid_error;
	
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
	
	bool flip = sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1];
	my_memset(dst, 0, 8);
	WriteDiff(dst, use_differential[!!flip]);
	WriteFlip(dst, flip);
	
	uchar sub_block_off_0 = flip ? 2 : 0;
	uchar sub_block_off_1 = sub_block_off_0 + 1;
	
	if (use_differential[!!flip]) {
		WriteColors555(dst, &sub_block_avg[sub_block_off_0], &sub_block_avg[sub_block_off_1]);
	} else {
		WriteColors444(dst, &sub_block_avg[sub_block_off_0], &sub_block_avg[sub_block_off_1]);
	}
	
	unsigned long lumi_error1 = computeLuminance(dst, &sub_block_src[sub_block_off_0], &sub_block_avg[sub_block_off_0], 0, g_idx_to_num[sub_block_off_0], threshold);
	unsigned long lumi_error2 = computeLuminance(dst, &sub_block_src[sub_block_off_1], &sub_block_avg[sub_block_off_1], 1, g_idx_to_num[sub_block_off_1], threshold);
	
	return lumi_error1 + lumi_error2;
}

/**
 * @brief OpenCL Kernel for parallel ETC1 texture compression.
 * 
 * Each work-item processes one 4x4 block of the input texture.
 * 
 * @param width Texture width.
 * @param height Texture height.
 * @param src Global pointer to the source BGRA texture data.
 * @param dst Global pointer to the destination buffer for compressed blocks.
 * @param ans Global pointer to accumulate the total compression error.
 */
__kernel void mmul(const int width,	const int height,
					  __global uchar* src,
					  __global uchar* dst,
					  __global unsigned long* ans) 
{
	int y = get_global_id(0) * 4;
	int x = get_global_id(1) * 4;

	int offset_src = y * width * 4 + x * 4;
	int offset_dst = (get_global_id(0) * (width / 4) + get_global_id(1)) * 8;

	union Color ver_blocks[16];
	union Color hor_blocks[16];

	__global const union Color* row0 = (__global union Color*)(src + offset_src);
	__global const union Color* row1 = row0 + width;
	__global const union Color* row2 = row1 + width;
	__global const union Color* row3 = row2 + width;
	
	// Prepare sub-blocks for vertical and horizontal split testing.
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
	
	atomic_add(ans, compressBlock(dst + offset_dst, ver_blocks, hor_blocks, INT32_MAX));
}

>>>> file: texture_compress_skl.cpp
/**
 * @file texture_compress_skl.cpp
 * @brief Host-side implementation of the TextureCompressor class.
 * 
 * Handles OpenCL initialization, resource management, and kernel execution for
 * ETC1 texture compression.
 */

#include "compress.hpp"
#include <iostream>
#include <vector>
#include <string>
#include "helper.hpp"

using namespace std;

/**
 * @brief Constructor: Initializes OpenCL environment.
 * 
 * Discovers available platforms and devices, selects a GPU, creates an OpenCL
 * context and command queue, and compiles the compression kernel.
 */
TextureCompressor::TextureCompressor() { 
	cl_platform_id platform;
	cl_uint platform_num = 0;
	cl_uint device_num = 0;
	size_t attr_size = 0;
	cl_char* attr_data = NULL;

	CL_ERR( clGetPlatformIDs(0, NULL, &platform_num));
	this->platform_ids = new cl_platform_id[platform_num];
	CL_ERR( clGetPlatformIDs(platform_num, this->platform_ids, NULL));

	for(uint platf=0; platf<platform_num; platf++) {
		platform = platform_ids[platf];
		clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &device_num);
		
		if (device_num) {
			device_ids = new cl_device_id[device_num];
			CL_ERR( clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, device_num, device_ids, NULL));
			device = device_ids[0];
			break;
		}
	}

	int ret;
	this->context = clCreateContext(0, 1, &this->device, NULL, NULL, &ret);
	CL_ERR( ret );
	
	this->command_queue = clCreateCommandQueue(this->context, this->device, CL_QUEUE_PROFILING_ENABLE, &ret);
	CL_ERR( ret );
		
	string kernel_src;
	read_kernel("sol_device.cl", kernel_src);
	const char* kernel_c_str = kernel_src.c_str();
	
	this->program = clCreateProgramWithSource(context, 1, (const char **) &kernel_c_str, NULL, &ret);
	CL_ERR( ret );
	
	ret = clBuildProgram(this->program, 0, NULL, NULL, NULL, NULL);
	CL_COMPILE_ERR( ret, this->program, this->device );
	
	this->kernel = clCreateKernel(program, "mmul", &ret);
	CL_ERR( ret );
}

/**
 * @brief Destructor: Releases OpenCL resources.
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
 * @brief Compresses a texture using the initialized OpenCL kernel.
 * 
 * Orchestrates memory transfers between host and device, executes the kernel,
 * and retrieves the compressed data and error metric.
 * 
 * @param src Pointer to source BGRA texture data.
 * @param dst Pointer to destination buffer for compressed data.
 * @param width Texture width.
 * @param height Texture height.
 * @return unsigned long Total compression error metric.
 */
unsigned long TextureCompressor::compress(const uint8_t* src,
									  uint8_t* dst,
									  int width,
									  int height)
{
	size_t global[2];
	cl_mem src_mem, dst_mem, ans_mem;
	unsigned long ans = 0;
	int sz = width * height;

	src_mem  = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(uint8_t) * sz * 4, NULL, NULL);
	dst_mem  = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(uint8_t) * sz / 2, NULL, NULL);
	ans_mem  = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(unsigned long) , NULL, NULL);

	clSetKernelArg(kernel, 0, sizeof(int), &width);
	clSetKernelArg(kernel, 1, sizeof(int), &height);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &src_mem);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &dst_mem);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), &ans_mem);

	clEnqueueWriteBuffer(this->command_queue, src_mem, CL_TRUE, 0, sizeof(uint8_t) * sz * 4, src, 0, NULL, NULL);
	clEnqueueWriteBuffer(this->command_queue, ans_mem, CL_TRUE, 0, sizeof(unsigned long), &ans, 0, NULL, NULL);

	global[0] = height / 4;
	global[1] = width / 4;
	
	clEnqueueNDRangeKernel(this->command_queue, kernel, 2, NULL, global, NULL, 0, NULL, NULL);
	clFinish(this->command_queue);
	
	clEnqueueReadBuffer(this->command_queue, dst_mem, CL_TRUE, 0, sizeof(uint8_t) * sz / 2, dst, 0, NULL, NULL);
	clEnqueueReadBuffer(this->command_queue, ans_mem, CL_TRUE, 0, sizeof(unsigned long), &ans, 0, NULL, NULL);

	clReleaseMemObject(src_mem);
	clReleaseMemObject(dst_mem);
	clReleaseMemObject(ans_mem);

	return ans;
}
