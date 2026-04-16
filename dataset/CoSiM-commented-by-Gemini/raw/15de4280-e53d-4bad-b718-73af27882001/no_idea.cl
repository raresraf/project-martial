/**
 * @file no_idea.cl
 * @brief OpenCL implementation of an ETC1-like texture compression algorithm.
 * @details This file contains the kernel and helper functions for compressing a 4x4
 * pixel block into the ETC1 format. The algorithm works by dividing a 4x4 block
 * into two sub-blocks and encoding each with a base color and a set of modifier
 * values chosen from a codeword table to minimize color error. It supports
 * differential mode (where sub-block colors are offsets from each other) and
 * individual mode, as well as a "flip" bit to change the sub-block orientation
 * from two 4x2 blocks to two 2x4 blocks.
 */

// Represents a 32-bit BGRA color.
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

// Clamps a uchar value to a given range.
uchar clampa(uchar val, uchar min, uchar max) {
	return val  max ? max : val);
}

// Clamps an int value to a given range.
int clampa_int(int val, int min, int max) {
	return val  max ? max : val);
}

// Rounds a 0-255 float value to a 5-bit representation (0-31).
uchar round_to_5_bits(float val) {
	return clampa (val * 31.0f / 255.0f + 0.5f, 0, 31);
}

// Rounds a 0-255 float value to a 4-bit representation (0-15).
uchar round_to_4_bits(float val) {
	return clampa (val * 15.0f / 255.0f + 0.5f, 0, 15);
}

// Codeword tables for intensity modifications in ETC1. Each table provides
// four modifier values for pixel luminance.
__constant short g_codeword_tables[8][4] = {
	{-8, -2, 2, 8},
	{-17, -5, 5, 17},
	{-29, -9, 9, 29},
	{-42, -13, 13, 42},
	{-60, -18, 18, 60},
	{-80, -24, 24, 80},
	{-106, -33, 33, 106},
	{-183, -47, 47, 183}};

// Maps a 2-bit modifier index to a 2-bit pixel index.
__constant uchar g_mod_to_pix[4] = {3, 2, 0, 1};

// Maps a linear texel index (0-7) within a sub-block to its position in the
// final 32-bit pixel data word, for both horizontal and vertical splits.
__constant uchar g_idx_to_num[4][8] = {
	{0, 4, 1, 5, 2, 6, 3, 7},        // Sub-block 0, no flip
	{8, 12, 9, 13, 10, 14, 11, 15},  // Sub-block 1, no flip
	{0, 4, 8, 12, 1, 5, 9, 13},      // Sub-block 0, flip
	{2, 6, 10, 14, 3, 7, 11, 15}     // Sub-block 1, flip
};

// Applies a luminance modifier to a base color.
union Color makeColor(union Color base, short lum) {
	int b = (int) base.channels.b + lum;
	int g = (int) base.channels.g + lum;
	int r = (int) base.channels.r + lum;
	union Color color;
	color.channels.b = (uchar) clampa_int(b, 0, 255);
	color.channels.g = (uchar) clampa_int(g, 0, 255);
	color.channels.r = (uchar) clampa_int(r, 0, 255);
	return color;
}


// Writes the packed 4-bit base colors for individual mode.
void WriteColors444(__global uchar* block,
						   const union Color color0,
						   const union Color color1) {
	block[0] = (color0.channels.r & 0xf0) | (color1.channels.r >> 4);
	block[1] = (color0.channels.g & 0xf0) | (color1.channels.g >> 4);
	block[2] = (color0.channels.b & 0xf0) | (color1.channels.b >> 4);
}

// Table for converting a 3-bit signed delta into a 3-bit two's complement representation.
__constant uchar two_compl_trans_table[8] = {
		4, 5, 6, 7, 0, 1, 2, 3,
	};

// Writes the 5-bit base color and 3-bit delta for differential mode.
void WriteColors555(__global uchar* block,
						   const union Color color0,
						   const union Color color1) {
	__private short delta_r = (short) ((color1.channels.r >> 3) - (color0.channels.r >> 3));
	__private short delta_g = (short) ((color1.channels.g >> 3) - (color0.channels.g >> 3));
	__private short delta_b = (short) ((color1.channels.b >> 3) - (color0.channels.b >> 3));
	
	block[0] = (color0.channels.r & 0xf8) | two_compl_trans_table[delta_r + 4];
	block[1] = (color0.channels.g & 0xf8) | two_compl_trans_table[delta_g + 4];
	block[2] = (color0.channels.b & 0xf8) | two_compl_trans_table[delta_b + 4];
}

// Writes the 3-bit codeword table index for a given sub-block.
void WriteCodewordTable(__global uchar* block,
							   uchar sub_block_id,
							   uchar table) {
	uchar shift = (2 + (3 - sub_block_id * 3));
	block[3] &= ~(0x07 << shift);
	block[3] |= table << shift;
}

// Writes the 32-bit packed pixel indices to the block.
void WritePixelData(__global uchar* block, uint pixel_data) {
	block[4] |= pixel_data >> 24;
	block[5] |= (pixel_data >> 16) & 0xff;
	block[6] |= (pixel_data >> 8) & 0xff;
	block[7] |= pixel_data & 0xff;
}

// Writes the flip bit to the block.
void WriteFlip(__global uchar* block, char flip) {
	block[3] &= ~0x01;
	block[3] |= flip;
}

// Writes the differential mode bit to the block.
void WriteDiff(__global uchar* block, char diff) {
	block[3] &= ~0x02;
	block[3] |= (char) diff << 1;
}

// Extracts a 4x4 block of pixels from a larger source image.
void ExtractBlock(uchar* dst, uchar* src, int width) {
	for (int j = 0; j < 4; ++j) {
		for ( int i = 0 ; i < 16; i++)
			dst[j * 4 * 4 + i ] = src[i];
		src += width * 4;
	}
}



// Creates a color quantized to 4 bits per channel, then expanded back to 8 bits.
union Color makeColor444(float* bgr) {
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


// Creates a color quantized to 5 bits per channel, then expanded back to 8 bits.
union Color makeColor555(float* bgr) {
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

// Calculates the squared Euclidean distance between two colors.
int getColorError(const union Color u, const union Color v) {
	int delta_b = (int) u.channels.b - v.channels.b;
	int delta_g = (int) u.channels.g - v.channels.g;
	int delta_r = (int) u.channels.r - v.channels.r;
	return delta_b * delta_b + delta_g * delta_g + delta_r * delta_r;
}

// Calculates the average color of an 8-pixel sub-block.
void getAverageColor(const union Color* src, float* avg_color)
{
	uint sum_b = 0, sum_g = 0, sum_r = 0;
	for (unsigned int i = 0; i < 8; ++i) {
		sum_b += src[i].channels.b;
		sum_g += src[i].channels.g;
		sum_r += src[i].channels.r;
	}
	float kInv8 = 1.0f / 8.0f;
	avg_color[0] = ((float)sum_b) * kInv8;
	avg_color[1] = ((float)sum_g) * kInv8;
	avg_color[2] = ((float)sum_r) * kInv8;
}

/**
 * @brief Finds the best codeword table and modifier indices for a sub-block.
 * @param block The destination compressed block.
 * @param src The 8-pixel source sub-block.
 * @param base The base color for the sub-block.
 * @param sub_block_id The ID of the sub-block (0 or 1).
 * @param idx_to_num_tab The lookup table for pixel ordering.
 * @param threshold The error threshold for early termination.
 * @return The total error for the best encoding found.
 * @details This function iterates through all 8 codeword tables. For each table, it
 * generates 4 candidate colors by applying the table's modifiers to the base
 * color. It then finds the best candidate for each of the 8 source pixels,
 * accumulating the total error. The table that results in the minimum total
 * error is chosen, and the corresponding table index and pixel indices are
 * written to the destination block.
 */
unsigned long computeLuminance(__global uchar* block,
						   const union Color* src,
						   const union Color base,
						   int sub_block_id,
						   __constant uchar* idx_to_num_tab,
						   unsigned long threshold)
{
	uint best_tbl_err = threshold;
	uchar best_tbl_idx = 0;
	uchar best_mod_idx[8][8];

	// Iterate through all possible codeword tables to find the one with the minimum error.
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		// Generate the 4 candidate colors for the current table.
		union Color candidate_color[4];
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			candidate_color[mod_idx] = makeColor(base, lum);
		}
		
		uint tbl_err = 0;
		// For each pixel in the sub-block, find the best modifier index.
		for (unsigned int i = 0; i < 8; ++i) {
			uint best_mod_err = threshold;
			for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
				const union Color *color = &candidate_color[mod_idx];
				uint mod_err = getColorError(src[i], *color);
				if (mod_err < best_mod_err) {
					best_mod_idx[tbl_idx][i] = mod_idx;
					best_mod_err = mod_err;
					if (mod_err == 0) break; // Early exit if perfect match
				}
			}
			tbl_err += best_mod_err;
			if (tbl_err > best_tbl_err) break; // Early exit if error exceeds best found
		}
		
		if (tbl_err < best_tbl_err) {
			best_tbl_err = tbl_err;
			best_tbl_idx = tbl_idx;
			if (tbl_err == 0) break; // Early exit if perfect match
		}
	}

	// Write the best found table index to the compressed block.
	WriteCodewordTable(block, sub_block_id, best_tbl_idx);

	// Pack the modifier indices for each pixel into a 32-bit word.
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
 * @brief Attempts to compress a 4x4 block as a single solid color.
 * @param dst The destination compressed block.
 * @param src The 16-pixel source block.
 * @param error Pointer to store the resulting compression error.
 * @return 1 if the block was successfully compressed as solid, 0 otherwise.
 * @details This is a fast-path optimization. It checks if all pixels in the
 * block are identical. If so, it encodes the block in differential mode with
 * two identical base colors and finds the best modifier to represent that
 * single color, applying it to all 16 pixels.
 */
int tryCompressSolidBlock(__global uchar* dst,
						   const union Color* src,
						   unsigned long* error)
{
	for (unsigned int i = 1; i < 16; ++i) {
		if (src[i].bits != src[0].bits)
			return 0; // Not a solid color block
	}
	
	for( int i = 0 ; i < 8; i++) dst[i] = 0;
	
	// Use differential mode (diff=1) with identical base colors.
	float src_color_float[3];
	src_color_float[0] = (float) src->channels.b;
	src_color_float[1] = (float) src->channels.g;
	src_color_float[2] = (float) src->channels.r;
	union Color base = makeColor555(src_color_float);
	
	WriteDiff(dst, 1);
	WriteFlip(dst, 0);
	WriteColors555(dst, base, base);
	
	// Find the best single modifier for the solid color.
	uchar best_tbl_idx = 0;
	uchar best_mod_idx = 0;
	int best_mod_err = INT_MAX; 
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			const union Color color = makeColor(base, lum);
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
	
	// Apply the same modifier to all pixels.
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
	return 1;
}

/**
 * @brief Compresses a 4x4 pixel block into ETC1 format.
 * @param dst The destination for the 8-byte compressed block.
 * @param ver_src Source pixels arranged for vertical sub-block processing.
 * @param hor_src Source pixels arranged for horizontal sub-block processing.
 * @param threshold Error threshold for early termination.
 * @return The total compression error for the block.
 * @details This function orchestrates the compression. It first checks for the
 * solid color case. If not, it calculates average colors for both horizontal
 * and vertical splits to decide on flip mode and differential/individual modes.
 * It then calls computeLuminance for the two chosen sub-blocks.
 */
unsigned long compressBlock(__global uchar* dst,
							union Color* ver_src,
							union Color* hor_src,
							unsigned long threshold)
{
	unsigned long solid_error = 0;
	// Attempt fast path for solid color blocks.
	if (tryCompressSolidBlock(dst, ver_src, &solid_error)) {
		return solid_error;
	}
	
	const union Color *sub_block_src[4] = {ver_src, ver_src + 8, hor_src, hor_src + 8};
	
	union Color sub_block_avg[4];
	char use_differential[2] = {1, 1};
	
	// Determine whether to use differential mode for each potential split (horizontal/vertical).
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
			
			// If color difference is too large, use individual (444) mode.
			int component_diff = v - u;
			if (component_diff  3) {
				use_differential[i / 2] = false;
				sub_block_avg[i] = makeColor444(avg_color_0);
				sub_block_avg[j] = makeColor444(avg_color_1);
			} else {
				sub_block_avg[i] = avg_color_555_0;
				sub_block_avg[j] = avg_color_555_1;
			}
		}
	}
	
	// Calculate approximation errors for both horizontal and vertical splits.
	uint sub_block_err[4] = {0};
	for (unsigned int i = 0; i < 4; ++i) {
		for (unsigned int j = 0; j < 8; ++j) {
			sub_block_err[i] += getColorError(sub_block_avg[i], sub_block_src[i][j]);
		}
	}
	
	// Choose the flip mode (horizontal vs. vertical split) that has less error.
	char flip =
	sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1];
	
	for (int i = 0 ; i < 8 ; i++) dst[i] = 0;
	
	// Write the header bits (diff and flip).
	WriteDiff(dst, use_differential[!!flip]);
	WriteFlip(dst, flip);
	
	uchar sub_block_off_0 = flip ? 2 : 0;
	uchar sub_block_off_1 = sub_block_off_0 + 1;
	
	// Write the base colors for the chosen mode.
	if (use_differential[!!flip]) {
		WriteColors555(dst, sub_block_avg[sub_block_off_0],
					   sub_block_avg[sub_block_off_1]);
	} else {
		WriteColors444(dst, sub_block_avg[sub_block_off_0],
					   sub_block_avg[sub_block_off_1]);
	}
	
	unsigned long lumi_error1 = 0, lumi_error2 = 0;
	
	// Compress the two sub-blocks.
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

// Simple memcpy implementation for use within the kernel.
void memcpy(uchar *dst, __global uchar *src, int size){
	for (int i = 0 ; i < size; i++)
		dst[i] = src[i];
}


/**
 * @brief Main OpenCL kernel for ETC1 texture compression.
 * @param width The width of the source image.
 * @param height The height of the source image.
 * @param src The global memory buffer for the source image data (BGRA).
 * @param dst The global memory buffer for the compressed output data.
 * @details Each work-item in the NDRange is responsible for compressing a
 * single 4x4 block of pixels. It reads the block, prepares the data for
 * both horizontal and vertical sub-block splits, and calls `compressBlock`
 * to perform the compression and write the 8-byte result.
 */
__kernel void copy_image(const int width, const int height,
				__global uchar *src,
				__global uchar *dst)
{
	int col = get_global_id(0);
	int linie = get_global_id(1);
	unsigned long compressed_error = 0;

	long i;
	union Color ver_blocks[16];
	union Color hor_blocks[16];
	
	// Calculate pointer to the start of the 4x4 source block for this work-item.
	__global union Color* row0 = (__global union Color*) (src + 4 * linie * (width * 4) + col * 4 * 4);
	__global union Color* row1 = row0 + width;
	__global union Color* row2 = row1 + width;
	__global union Color* row3 = row2 + width;
	
	// Rearrange source pixels into memory layouts for both vertical and horizontal splits.
	memcpy((char*)&ver_blocks[0], row0, 8);
	memcpy((char*)&ver_blocks[2], row1, 8);
	memcpy((char*)&ver_blocks[4], row2, 8);
	memcpy((char*)&ver_blocks[6], row3, 8);
	memcpy((char*)&ver_blocks[8], &row0[2], 8);
	memcpy((char*)&ver_blocks[10], &row1[2], 8);
	memcpy((char*)&ver_blocks[12], &row2[2], 8);
	memcpy((char*)&ver_blocks[14], &row3[2], 8);
	
	memcpy((char*)&hor_blocks[0], row0, 16);
	memcpy((char*)&hor_blocks[4], row1, 16);
	memcpy((char*)&hor_blocks[8], row2, 16);
	memcpy((char*)&hor_blocks[12], row3, 16);
	
	// Compress the block and write the result to the destination buffer.
	compressed_error += compressBlock(dst + 8 * (width/4*linie+col), ver_blocks, hor_blocks, INT_MAX);
}

// The following is C++ host code for setting up and running the OpenCL kernel.
// It seems to have been concatenated into the same file.

/**********************************************************************************
/* @file texture_compress_skl.cpp
/* @brief C++ host code for orchestrating OpenCL-based texture compression.
/* @details This code is responsible for setting up the OpenCL environment,
/* loading and compiling the OpenCL kernel, creating memory buffers, executing
/* the compression kernel, and reading back the results.
/**********************************************************************************

#include "compress.hpp"

using namespace std;

// Macro for dying on an assertion failure, printing error and line number.
#define DIE(assertion, call_description)         \
	do                                           \
	{                                            \
		if (assertion)                           \
		{                                        \
			fprintf(stderr, "(%d): ", __LINE__); \
			perror(call_description);            \
			exit(EXIT_FAILURE);                  \
		}                                        \
	} while (0);

// Converts an OpenCL error code to a human-readable string.
const char *cl_get_string_err(cl_int err)
{
	switch (err)
	{
	case CL_SUCCESS: return "Success!";
	case CL_DEVICE_NOT_FOUND: return "Device not found.";
	case CL_DEVICE_NOT_AVAILABLE: return "Device not available";
	case CL_COMPILER_NOT_AVAILABLE: return "Compiler not available";
	// ... (many other cases) ...
	default: return "Unknown";
	}
}

// Retrieves and prints the build log for a failed OpenCL program compilation.
void cl_get_compiler_err_log(cl_program program, cl_device_id device)
{
	char *build_log;
	size_t log_size;
	
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
	build_log = new char[log_size + 1];
	
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
	build_log[log_size] = '\0';
	cout << endl << build_log << endl;
}

// Checks for an OpenCL error and prints a message if one occurred.
int CL_ERR(int cl_ret)
{
	if (cl_ret != CL_SUCCESS) {
		cout << endl << cl_get_string_err(cl_ret) << endl;
		return 1;
	}
	return 0;
}

// Checks for an OpenCL compilation error and prints the build log if one occurred.
int CL_COMPILE_ERR(int cl_ret, cl_program program, cl_device_id device)
{
	if (cl_ret != CL_SUCCESS) {
		cout << endl << cl_get_string_err(cl_ret) << endl;
		cl_get_compiler_err_log(program, device);
		return 1;
	}
	return 0;
}

// Reads an OpenCL kernel file into a string.
void read_kernel(string file_name, string &str_kernel)
{
	ifstream in_file(file_name.c_str());
	DIE(!in_file.is_open(), "ERR OpenCL kernel file. Same directory as binary ?");

	stringstream str_stream;
	str_stream << in_file.rdbuf();
	str_kernel = str_stream.str();
}

// Finds the first available GPU device on the system.
void gpu_find(cl_device_id &device)
{
	cl_platform_id platform;
	cl_uint platform_num = 0;
	cl_platform_id *platform_list = NULL;
	cl_uint device_num = 0;
	cl_device_id *device_list = NULL;
	
	// Discover platforms and devices
	CL_ERR(clGetPlatformIDs(0, NULL, &platform_num));
	platform_list = new cl_platform_id[platform_num];
	DIE(platform_list == NULL, "alloc platform_list");
	CL_ERR(clGetPlatformIDs(platform_num, platform_list, NULL));
	cout << "Platforms found: " << platform_num << endl;
	int gpu_platform = -1;
	
	// Iterate through platforms to find a GPU
	for (uint platf = 0; platf < platform_num; platf++)
	{
		CL_ERR(clGetDeviceIDs(platform_list[platf], CL_DEVICE_TYPE_GPU, 0, NULL, &device_num));
		if (device_num == 0) continue;
		gpu_platform = platf;
		device_list = new cl_device_id[device_num];
		DIE(device_list == NULL, "alloc devices");
		CL_ERR(clGetDeviceIDs(platform_list[platf], CL_DEVICE_TYPE_GPU, device_num, device_list, NULL));
		cout << "\tDevices found " << device_num << endl;
		device = device_list[0];
		break;
	}
	cout << "Selected device " << device << " from platform " << gpu_platform << endl;
	delete[] platform_list;
	delete[] device_list;
}

// Constructor for the TextureCompressor, finds the GPU device.
TextureCompressor::TextureCompressor()
{
	gpu_find(device);
}
TextureCompressor::~TextureCompressor() {}

/**
 * @brief Compresses a texture using the OpenCL kernel.
 * @param src Pointer to the source image data.
 * @param dst Pointer to the destination buffer for compressed data.
 * @param width The width of the image.
 * @param height The height of the image.
 * @return 0 on success.
 * @details This method orchestrates the entire compression process:
 * 1. Creates an OpenCL context and command queue.
 * 2. Allocates device memory buffers for source and destination data.
 * 3. Copies the source data to the device.
 * 4. Reads, builds, and compiles the OpenCL kernel from "no_idea.cl".
 * 5. Sets the kernel arguments.
 * 6. Enqueues the NDRange kernel for execution across all 4x4 blocks.
 * 7. Reads the compressed data back from the device.
 * 8. Cleans up all OpenCL resources.
 */
unsigned long TextureCompressor::compress(const uint8_t* src,
									  uint8_t* dst,
									  int width,
									  int height)
{
	cl_int ret;
	string kernel_src;
	
	// Setup OpenCL context and command queue
	context = clCreateContext(0, 1, &device, NULL, NULL, &ret);
	CL_ERR(ret);
	command_queue = clCreateCommandQueue(context, device, 0, &ret);
	CL_ERR(ret);

	// Create device buffers and copy host data
	cl_mem src_dev = clCreateBuffer(context, CL_MEM_READ_WRITE, 4 * width * height, NULL, &ret);
	CL_ERR(ret);
	cl_mem dst_dev = clCreateBuffer(context, CL_MEM_READ_WRITE, width * height * 4 / 8, NULL, &ret);
	CL_ERR(ret);
	ret = clEnqueueWriteBuffer(command_queue, src_dev, CL_TRUE, 0, 4 * width * height, src, 0, NULL, NULL);
	CL_ERR(ret);

	// Read and compile the kernel
	read_kernel("no_idea.cl", kernel_src);
	const char *kernel_c_str = kernel_src.c_str();
	program = clCreateProgramWithSource(context, 1, &kernel_c_str, NULL, &ret);
	CL_ERR(ret);
	ret = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	CL_COMPILE_ERR(ret, program, device);

	// Create the kernel and set its arguments
	kernel = clCreateKernel(program, "copy_image", &ret);
	CL_ERR(ret);
	CL_ERR(clSetKernelArg(kernel, 0, sizeof(cl_uint), (void *)&width));
	CL_ERR(clSetKernelArg(kernel, 1, sizeof(cl_uint), (void *)&height));
	CL_ERR(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&src_dev));
	CL_ERR(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&dst_dev));
	
	// Define the global work size (one work-item per 4x4 block)
	size_t globalSize[2] = {width/4,height/4};
	
	// Execute the kernel
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, globalSize, 0, 0, NULL, NULL);
	CL_ERR(ret);
	
	// Wait for completion and read results
	CL_ERR(clFinish(command_queue));
	CL_ERR(clEnqueueReadBuffer(command_queue, dst_dev, CL_TRUE, 0, width * height * 4 / 8, dst, 0, NULL, NULL));

	// Release OpenCL resources
	CL_ERR(clReleaseMemObject(src_dev));
	CL_ERR(clReleaseMemObject(dst_dev));
	CL_ERR(clReleaseCommandQueue(command_queue));
	CL_ERR(clReleaseContext(context));
	
	cout << " DONE\n ";
	return 0;
}
