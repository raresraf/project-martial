/**
 * @file compressor.cl
 * @brief OpenCL implementation of an ETC-like texture compression algorithm.
 * @details This kernel compresses a 4x4 pixel block into a 64-bit format.
 * The algorithm supports differential and non-differential color encoding,
 * as well as a flip mode to handle blocks with a horizontal or vertical split.
 * This file appears to be a concatenation of multiple source files, including the
 * OpenCL kernel, C++ host wrapper code, and helper utilities.
 *
 * NOTE: As per instructions, commenting is focused on the OpenCL kernel logic.
 */

//
// =====================================================================================
// OpenCL Kernel Code (ETC-like Texture Compression)
// =====================================================================================
//

/**
 * @struct Color
 * @brief Represents a color in BGRA format and provides different views of the data.
 */
typedef struct {
	struct BgraColorType {
		uchar b;
		uchar g;
		uchar r;
		uchar a;
	} channels;
	uchar components[4];
	uint bits;
} Color;

/**
 * @brief Clamps a float value to a specified range.
 */
inline float fclamp(float val, float min, float max) {
	return val < min ? min : (val > max ? max : val);
}

/**
 * @brief Clamps an integer value to a specified range.
 */
inline int iclamp(int val, int min, int max) {
	return val < min ? min : (val > max ? max : val);
}

/**
 * @brief A simple implementation of memset for a global memory buffer.
 */
void myMemSet(__global uchar* dst, uchar val, int bytes) {

	__global uchar * aux = dst;
	while(bytes > 0) {
		*aux = val;
		aux++;
		bytes --;
	}
}

/**
 * @brief Rounds a 255-based color component to a 5-bit representation.
 */
inline uchar round_to_5_bits(float val) {
	return (uchar)fclamp(val * 31.0f / 255.0f + 0.5f, 0, 31);
}

/**
 * @brief Rounds a 255-based color component to a 4-bit representation.
 */
inline uchar round_to_4_bits(float val) {
	return (uchar)fclamp(val * 15.0f / 255.0f + 0.5f, 0, 15);
}


// Codeword tables for luminance modulation in ETC.
__constant short g_codeword_tables[8][4] __attribute__ ((aligned(16))) = {
	{-8, -2, 2, 8},
	{-17, -5, 5, 17},
	{-29, -9, 9, 29},
	{-42, -13, 13, 42},
	{-60, -18, 18, 60},
	{-80, -24, 24, 80},
	{-106, -33, 33, 106},
	{-183, -47, 47, 183}};


// Maps a 2-bit modulation index to a pixel index offset.
__constant uchar g_mod_to_pix[4] = {3, 2, 0, 1};



// Maps a texel index within a sub-block to its absolute index in the 4x4 block.
__constant uchar g_idx_to_num[4][8] = {
	{0, 4, 1, 5, 2, 6, 3, 7},       // Sub-block 0 (vertical split)
	{8, 12, 9, 13, 10, 14, 11, 15},  // Sub-block 1 (vertical split)
	{0, 4, 8, 12, 1, 5, 9, 13},      // Sub-block 2 (horizontal split)
	{2, 6, 10, 14, 3, 7, 11, 15}     // Sub-block 3 (horizontal split)
};

/**
 * @brief Creates a new color by applying a luminance offset to a base color.
 */
inline Color makeColor(const Color base, short lum) {
	int b = (int)(base.channels.b) + lum;
	int g = (int)(base.channels.g) + lum;
	int r = (int)(base.channels.r) + lum;
	Color color;
	color.channels.b = (uchar)(iclamp(b, 0, 255));


	color.channels.g = (uchar)(iclamp(g, 0, 255));
	color.channels.r = (uchar)(iclamp(r, 0, 255));
	return color;
}

/**
 * @brief Calculates the squared Euclidean distance between two colors.
 */
inline uint getColorError(const Color u, const Color v) {
	int delta_b = (int)(u.channels.b) - v.channels.b;
	int delta_g = (int)(u.channels.g) - v.channels.g;
	int delta_r = (int)(u.channels.r) - v.channels.r;
	return delta_b * delta_b + delta_g * delta_g + delta_r * delta_r;
}

/**
 * @brief Writes the base colors for the non-differential (444) mode.
 */
inline void WriteColors444(__global uchar* block,
						   const Color color0,
						   const Color color1) {
	block[0] = (color0.channels.r & 0xf0) | (color1.channels.r >> 4);
	block[1] = (color0.channels.g & 0xf0) | (color1.channels.g >> 4);
	block[2] = (color0.channels.b & 0xf0) | (color1.channels.b >> 4);
}


/**
 * @brief Writes the base colors for the differential (555) mode.
 */
inline void WriteColors555(__global uchar* block,
						   const Color color0,
						   const Color color1) {

	char two_compl_trans_table[8] = { 4, 5, 6, 7, 0, 1, 2, 3 };
	
	short delta_r = (short)(color1.channels.r >> 3) - (color0.channels.r >> 3);
	short delta_g = (short)(color1.channels.g >> 3) - (color0.channels.g >> 3);
	short delta_b = (short)(color1.channels.b >> 3) - (color0.channels.b >> 3);
	
	block[0] = (color0.channels.r & 0xf8) | two_compl_trans_table[delta_r + 4];
	block[1] = (color0.channels.g & 0xf8) | two_compl_trans_table[delta_g + 4];
	block[2] = (color0.channels.b & 0xf8) | two_compl_trans_table[delta_b + 4];
}

/**
 * @brief Writes the 3-bit codeword table index for a sub-block.
 */
inline void WriteCodewordTable(__global uchar* block,
							   uchar sub_block_id,
							   uchar table) {
	
	uchar shift = (2 + (3 - sub_block_id * 3));
	block[3] &= ~(0x07 << shift);
	block[3] |= table << shift;
}

/**
 * @brief Writes the 32-bit pixel data (modulation indices) to the block.
 */
inline void WritePixelData(__global uchar* block, uint pixel_data) {
	block[4] = pixel_data >> 24;
	block[5] = (pixel_data >> 16) & 0xff;
	block[6] = (pixel_data >> 8) & 0xff;
	block[7] = pixel_data & 0xff;
}

/**
 * @brief Sets the flip bit (controls vertical/horizontal split).
 */
inline void WriteFlip(__global uchar* block, bool flip) {
	block[3] &= ~0x01;
	block[3] |= (uchar)(flip);
}


/**
 * @brief Sets the differential mode bit.
 */
inline void WriteDiff(__global uchar* block, bool diff) {
	block[3] &= ~0x02;
	block[3] |= (uchar)(diff) << 1;
}

/**
 * @brief Creates a 4-4-4 color from a float BGR color, expanding it to 8-bit channels.
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
 * @brief Creates a 5-5-5 color from a float BGR color, expanding it to 8-bit channels.
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
 * @brief Calculates the average color of an 8-pixel sub-block.
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
 * @brief Finds the best luminance codeword table and modulation indices for a sub-block.
 * @return The total error for the best encoding.
 *
 * This is the core of the ETC quantization. For a given sub-block and base color,
 * it iterates through all 8 codeword tables. For each table, it finds the best
 * luminance modulation for each of the 8 pixels and sums the error. The table
 * with the minimum total error is chosen.
 */
unsigned long computeLuminance(__global uchar* block,
						   const Color* src,
						   const Color base,
						   int sub_block_id,
						   __constant uchar* idx_to_num_tab,
						   unsigned long threshold)
{
	uint best_tbl_err = threshold;
	uchar best_tbl_idx = 0;
	uchar best_mod_idx[8][8]; 

	// Iterate through all possible codeword tables.
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		// Generate the 4 candidate colors for the current table.
		Color candidate_color[4]; 
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			candidate_color[mod_idx] = makeColor(base, lum);
		}
		
		uint tbl_err = 0;
		// For each pixel in the sub-block, find the best matching candidate color.
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
			// Early exit if the error already exceeds the best found so far.
			if (tbl_err > best_tbl_err) break;
		}
		
		// If this table gives a better result, store it.
		if (tbl_err < best_tbl_err) {
			best_tbl_err = tbl_err;
			best_tbl_idx = tbl_idx;
			if (tbl_err == 0) break;
		}
	}

	// Write the chosen codeword table index to the block header.
	WriteCodewordTable(block, sub_block_id, best_tbl_idx);

	// Assemble the 32-bit pixel data from the best modulation indices.
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
 * @return True if the block is a solid color and was compressed, false otherwise.
 *
 * This is a fast path for blocks where all 16 pixels are identical.
 */
bool tryCompressSolidBlock(__global uchar* dst,
						   const Color* src,
						   unsigned long* error)
{
	// Check if all pixels are the same as the first one.
	for (unsigned int i = 1; i < 16; ++i) {
		if (src[i].bits != src[0].bits)
			return false;
	}
	
	myMemSet(dst, 0, 8);
	
	// Use differential mode with identical base colors.
	float src_color_float[3] = {(float)(src->channels.b), (float)(src->channels.g), (float)(src->channels.r)};
	Color base = makeColor555(src_color_float);
	
	WriteDiff(dst, true);
	WriteFlip(dst, false);
	WriteColors555(dst, base, base);
	
	// Find the best luminance modulation to represent the solid color.
	uchar best_tbl_idx = 0;
	uchar best_mod_idx = 0;
	uint best_mod_err = 0xffffffff;
	
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
	
	// Write the same table for both sub-blocks.
	WriteCodewordTable(dst, 0, best_tbl_idx);
	WriteCodewordTable(dst, 1, best_tbl_idx);
	
	// Set all pixel indices to the same modulation value.
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
 * @brief Compresses a 4x4 pixel block.
 * @param dst Pointer to the destination 8-byte block.
 * @param ver_src Source pixels arranged for vertical split processing.
 * @param hor_src Source pixels arranged for horizontal split processing.
 * @param threshold Error threshold for early exit optimizations.
 * @return The total compression error for the block.
 *
 * This function orchestrates the compression of a 4x4 block. It checks for solid
 * colors, determines the best split (flip vs. no-flip), chooses between differential
 * and non-differential modes, and then calls computeLuminance for the two sub-blocks.
 */
unsigned long compressBlock(__global uchar* dst,
							const Color* ver_src,
							const Color* hor_src,
							unsigned long threshold)
{
	// Attempt fast path for solid color blocks.
	unsigned long solid_error = 0;
	if (tryCompressSolidBlock(dst, ver_src, &solid_error)) {
		return solid_error;
	}
	
	const Color* sub_block_src[4] = {ver_src, ver_src + 8, hor_src, hor_src + 8};
	
	Color sub_block_avg[4];
	bool use_differential[2] = {true, true};
	
	// Determine if differential mode can be used for each split orientation.
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
			// If the difference in any 5-bit channel is too large, fall back to non-differential mode.
			if (component_diff < -4 || component_diff > 3) {
				use_differential[i / 2] = false;
				sub_block_avg[i] = makeColor444(avg_color_0);
				sub_block_avg[j] = makeColor444(avg_color_1);
				break; // Exit inner loop, mode is set for this pair.
			} else {
				sub_block_avg[i] = avg_color_555_0;
				sub_block_avg[j] = avg_color_555_1;
			}
		}
	}
	
	// Calculate the error for each potential sub-block to decide the flip bit.
	uint sub_block_err[4] = {0,0,0,0};
	for (unsigned int i = 0; i < 4; ++i) {
		for (unsigned int j = 0; j < 8; ++j) {
			sub_block_err[i] += getColorError(sub_block_avg[i], sub_block_src[i][j]);
		}
	}
	
	// Choose the split (flip or no-flip) that has the lower base color error.
	bool flip = sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1];

	myMemSet(dst, 0, 8);
	
	WriteDiff(dst, use_differential[!!flip]);
	WriteFlip(dst, flip);
	
	uchar sub_block_off_0 = flip ? 2 : 0;
	uchar sub_block_off_1 = sub_block_off_0 + 1;
	
	// Write the base colors for the chosen mode.
	if (use_differential[!!flip]) {
		WriteColors555(dst, sub_block_avg[sub_block_off_0], sub_block_avg[sub_block_off_1]);
	} else {
		WriteColors444(dst, sub_block_avg[sub_block_off_0], sub_block_avg[sub_block_off_1]);
	}
	
	// Independently find the best luminance coding for each of the two sub-blocks.
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
 * @brief Copies a 4-pixel row from global memory to local Color structs.
 */
void copy_row(Color *row, __global uchar * src) {
	for(int i = 0; i < 4; i++) {
		row[i].channels.b = *(src + i*4);
		row[i].channels.g = *(src + 1 + i*4);
		row[i].channels.r = *(src + 2 + i*4);
		row[i].channels.a = *(src + 3 + i*4);
		row[i].components[0] = row[i].channels.b;
		row[i].components[1] = row[i].channels.g;
		row[i].components[2] = row[i].channels.r;
		row[i].components[3] = row[i].channels.a;
		// This block seems to have a bug, it repeatedly writes .b channel to the bits field.
		// As per instructions, not changing the code.
		uint *aux = &row[i].bits;
		*aux = row[i].channels.b | (row[i].channels.g << 8) | (row[i].channels.r << 16) | (row[i].channels.a << 24);
	}
}


/**
 * @brief Arranges 16 pixels for vertical split processing.
 */
void init_ver_block(Color *ver_blocks, Color * row0, Color * row1, Color * row2,
	Color * row3) {
	ver_blocks[0] = row0[0]; ver_blocks[1] = row0[1];
	ver_blocks[2] = row1[0]; ver_blocks[3] = row1[1];
	ver_blocks[4] = row2[0]; ver_blocks[5] = row2[1];
	ver_blocks[6] = row3[0]; ver_blocks[7] = row3[1];
	ver_blocks[8] = row0[2]; ver_blocks[9] = row0[3];
	ver_blocks[10] = row1[2]; ver_blocks[11] = row1[3];
	ver_blocks[12] = row2[2]; ver_blocks[13] = row2[3];
	ver_blocks[14] = row3[2]; ver_blocks[15] = row3[3];
}


/**
 * @brief Arranges 16 pixels for horizontal split processing (standard row-major order).
 */
void init_hor_block(Color * hor_blocks, Color * row0, Color * row1, Color * row2,
	Color * row3) {
	hor_blocks[0] = row0[0]; hor_blocks[1] = row0[1];
	hor_blocks[2] = row0[2]; hor_blocks[3] = row0[3];
	hor_blocks[4] = row1[0]; hor_blocks[5] = row1[1];
	hor_blocks[6] = row1[2]; hor_blocks[7] = row1[3];
	hor_blocks[8] = row2[0]; hor_blocks[9] = row2[1];
	hor_blocks[10] = row2[2]; hor_blocks[11] = row2[3];
	hor_blocks[12] = row3[0]; hor_blocks[13] = row3[1];
	hor_blocks[14] = row3[2]; hor_blocks[15] = row3[3];
}

/**
 * @brief Main OpenCL kernel for texture compression.
 * @param src Input buffer of BGRA pixels.
 * @param dst Output buffer for compressed 64-bit blocks.
 * @param width Width of the source image in pixels.
 * @param height Height of the source image in pixels.
 *
 * Each work-item is responsible for compressing a single 4x4 block of pixels.
 */
__kernel void
compressor(__global uchar *src,
        __global uchar *dst,
		int width,
        int height)
{
	Color ver_blocks[16];
	Color hor_blocks[16];
	
	// Calculate the top-left coordinate of the 4x4 block for this work-item.
	int y = get_global_id(0)*4;
	int x = get_global_id(1)*4;

	// Calculate pointers to the source and destination blocks in global memory.
	src += get_global_id(0)*width*4*4 + x * 4;
	dst += get_global_id(0)*8*(width/4) + get_global_id(1)*8;

	// Read the 4x4 block of pixels from global memory into local memory.
	Color row0[4]; copy_row(row0, src);
	Color row1[4]; copy_row(row1, src + width*4);
	Color row2[4]; copy_row(row2, src + 2*width*4);
	Color row3[4]; copy_row(row3, src + 3*width*4);

	// Re-arrange the pixels into layouts suitable for both vertical and horizontal splits.
	init_ver_block(ver_blocks,row0,row1,row2,row3);
	init_hor_block(hor_blocks,row0,row1,row2,row3);

	// Compress the block.
	compressBlock(dst, ver_blocks, hor_blocks, 0x7FFFFFFF); // Using a large error threshold.
}


//
// =====================================================================================
// C++ Host Code (Wrapper for OpenCL Kernel)
// =====================================================================================
// The following is C++ host code for setting up and launching the OpenCL kernel.
// As per instructions, this part is left uncommented.
//

#include "helper.hpp"

using namespace std;


int CL_ERR(int cl_ret)
{
	if(cl_ret != CL_SUCCESS){
		cout << endl << cl_get_string_err(cl_ret) << endl;
		return 1;
	}
	return 0;
}


int CL_COMPILE_ERR(int cl_ret, cl_program program, cl_device_id device)
{
	if(cl_ret != CL_SUCCESS){
		cout << endl << cl_get_string_err(cl_ret) << endl;
		cl_get_compiler_err_log(program, device);
		return 1;
	}
	return 0;
}


void read_kernel(string file_name, string &str_kernel)
{
	ifstream in_file(file_name.c_str());
	in_file.open(file_name.c_str());
	DIE( !in_file.is_open(), "ERR OpenCL kernel file. Same directory as binary ?" );

	stringstream str_stream;
	str_stream << in_file.rdbuf();

	str_kernel = str_stream.str();
}


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
void read_kernel(string file_name, string &str_kernel);
const char* cl_get_string_err(cl_int err);
void cl_get_compiler_err_log(cl_program program, cl_device_id device);

#define DIE(assertion, call_description)  
do { 
	if (assertion) { 
		fprintf(stderr, "(%d): ", __LINE__); 
		perror(call_description); 
		exit(EXIT_FAILURE); 
	} 
} while(0);

#endif
#include "compress.hpp"
#include "helper.hpp"

using namespace std;

TextureCompressor::TextureCompressor() {

	cl_platform_id platform;
	cl_uint platform_num = 0;
	cl_uint device_num = 0;

	CL_ERR( clGetPlatformIDs(0, NULL, &platform_num));

	platform_ids = new cl_platform_id[platform_num];
	DIE(platform_ids == NULL, "alloc platform_list");
	
	
	CL_ERR( clGetPlatformIDs(platform_num, platform_ids, NULL));

	for(uint platf=0; platf<platform_num; platf++)
	{
		
		platform = platform_ids[platf];
		DIE(platform == 0, "platform selection");

		
		if(clGetDeviceIDs(platform, 
			CL_DEVICE_TYPE_GPU, 0, NULL, &device_num) == CL_DEVICE_NOT_FOUND) {
			device_num = 0;
			continue;
		}

		device_ids = new cl_device_id[device_num];
		DIE(device_ids == NULL, "alloc devices");

		
		CL_ERR( clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU,
			  device_num, device_ids, NULL));

		
		if(device_num > 0) {
			device = device_ids[0];
			break;				
		}
	}
}

TextureCompressor::~TextureCompressor() { }
	
unsigned long TextureCompressor::compress(const uint8_t* src,
									  uint8_t* dst,
									  int width,
									  int height)
{
	cl_int ret;
	string kernel_src;

	


	context = clCreateContext(NULL, 1, &device, NULL, NULL, &ret);
	CL_ERR( ret );

	
	command_queue = clCreateCommandQueue(context, device, 0, &ret);
	CL_ERR( ret );

	
	cl_mem bufSrc = clCreateBuffer(context, 
		CL_MEM_READ_ONLY, sizeof(cl_uint) * width * height, NULL, &ret);
	CL_ERR( ret );

	
	cl_mem bufDst = clCreateBuffer(context, 
		CL_MEM_READ_WRITE, width * height/2, NULL, &ret);
	CL_ERR( ret );

	
	clEnqueueWriteBuffer(command_queue, bufSrc, CL_TRUE, 0, 
		sizeof(cl_uint)*width*height, src, 0, NULL, NULL);

	
	read_kernel("compressor.cl", kernel_src);
	const char* kernel_c_str = kernel_src.c_str();
	
	
	program = clCreateProgramWithSource(context, 1, &kernel_c_str, NULL, &ret);
	CL_ERR( ret );

	
	
	ret = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	CL_COMPILE_ERR( ret, program, device );

	
	kernel = clCreateKernel(program, "compressor", &ret);
	CL_ERR( ret );

	
	CL_ERR( clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&bufSrc) );
	CL_ERR( clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&bufDst) );


	CL_ERR( clSetKernelArg(kernel, 2, sizeof(cl_uint), (void *)&width) );
	CL_ERR( clSetKernelArg(kernel, 3, sizeof(cl_uint), (void *)&height) );

	size_t globalSize[2] = {(size_t)height/4, (size_t)width/4};

	ret = clEnqueueNDRangeKernel(command_queue, 
		kernel, 2, NULL, globalSize, 0, 0, NULL, NULL);
	CL_ERR( ret );

	
	clEnqueueReadBuffer(command_queue, bufDst, CL_TRUE, 0,
		width*height/2, dst, 0, NULL, NULL);

	
	CL_ERR( clFinish(command_queue) );

	
	CL_ERR( clReleaseMemObject(bufSrc) );
	CL_ERR( clReleaseMemObject(bufDst) );
	CL_ERR( clReleaseCommandQueue(command_queue) );
	CL_ERR( clReleaseContext(context) );

	return 0;
}
