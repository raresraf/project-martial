/**
 * @file compress.cl
 * @brief OpenCL kernel for an ETC-like texture compression algorithm.
 *
 * This file implements a texture compression algorithm that shares similarities
 * with the Ericsson Texture Compression (ETC) format. It operates on 4x4 pixel
 * blocks and uses techniques like sub-block partitioning, differential color
 * encoding, and luminance modulation with codeword tables to achieve compression.
 * The kernel is designed to be executed on a GPU.
 */

// Defines for color channel indexing.
#define B_V 0
#define G_V 1
#define R_V 2
#define A_V 3

/**
 * @union Color
 * @brief Represents a 4-channel color (e.g., RGBA) that can be accessed
 *        either as individual uchar channels or as a single uint.
 */
typedef union color {
	uchar channels[4];
	uint bits;
} Color;

// Pre-defined codeword tables for luminance modification. Each table provides
// four intensity modifiers for a given base color.
__constant __attribute__((aligned(16))) short g_codeword_tables[8][4] = {
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

// Maps a linear texel index within a sub-block to its position in the final
// compressed block, for both vertical and horizontal splits.
__constant uchar g_idx_to_num[4][8] = {
	{0, 4, 1, 5, 2, 6, 3, 7},        
	{8, 12, 9, 13, 10, 14, 11, 15},  
	{0, 4, 8, 12, 1, 5, 9, 13},      
	{2, 6, 10, 14, 3, 7, 11, 15}     
};

// A simple GPU-side memcpy implementation.
void *
memcpy_gpu (void *dest, __global void *src, size_t len)
{
  char *d = (char*)dest;
  __global const char *s = (__global char*)src;
  while (len--)
    *d++ = *s++;
  return dest;
}

// Clamping functions for different data types.
int my_clampI(int val, int min, int max) {
	return val  max ? max : val);
}

float my_clampF(float val, float min, float max) {
	return val  max ? max : val);
}

uchar my_clampC(uchar val, uchar min, uchar max) {
	return val  max ? max : val);
}

// Rounds a float color component (0-255) to a 5-bit representation.
uchar round_to_5_bits(float val) {
	return my_clampC(val * 31.0f / 255.0f + 0.5f, 0, 31);
}

// Rounds a float color component (0-255) to a 4-bit representation.
uchar round_to_4_bits(float val) {
	return my_clampC(val * 15.0f / 255.0f + 0.5f, 0, 15);
}

/**
 * @brief Creates a new color by applying a luminance modification to a base color.
 * @param base The base color.
 * @param lum The luminance value to add to each channel.
 * @return The modified color.
 */
Color makeColor(const Color* base, short lum) {
	int b = (int)(base->channels[B_V]) + lum;
	int g = (int)(base->channels[G_V]) + lum;
	int r = (int)(base->channels[R_V]) + lum;
	Color color;
	color.channels[B_V] = (uchar)(my_clampI(b, 0, 255));
	color.channels[G_V] = (uchar)(my_clampI(g, 0, 255));
	color.channels[R_V] = (uchar)(my_clampI(r, 0, 255));
	return color;
}

/**
 * @brief Calculates the squared error between two colors.
 * @param u First color.
 * @param v Second color.
 * @return The squared Euclidean distance between the colors in RGB space.
 *         A perceptually weighted metric can be used if USE_PERCEIVED_ERROR_METRIC is defined.
 */
uint getColorError(const Color* u, const Color* v) {
#ifdef USE_PERCEIVED_ERROR_METRIC
	float delta_b = (float)(u->channels.[B_V]) - v->channels[B_V]
	float delta_g = (float)(u->channels.[G_V]) - v->channels[G_V]
	float delta_r = (float)(u->channels.[R_V]) - v->channels[R_V]

	return (uint)(0.299f * delta_b * delta_b +
								 0.587f * delta_g * delta_g +
								 0.114f * delta_r * delta_r);
#else
	int delta_b = (int)(u->channels[B_V]) - v->channels[B_V];
	int delta_g = (int)(u->channels[G_V]) - v->channels[G_V];
	int delta_r = (int)(u->channels[R_V]) - v->channels[R_V];

	return delta_b * delta_b + delta_g * delta_g + delta_r * delta_r;
#endif
}

/**
 * @brief Writes two 4-bit-per-channel colors to the compressed block.
 * @param block Pointer to the destination compressed block.
 * @param color0 The first color.
 * @param color1 The second color.
 */
void WriteColors444(__global uchar* block,
						   const Color* color0,
						   const Color* color1) {
	
	block[0] = (color0->channels[R_V] & 0xf0) | (color1->channels[R_V] >> 4);
	block[1] = (color0->channels[G_V] & 0xf0) | (color1->channels[G_V] >> 4);
	block[2] = (color0->channels[B_V] & 0xf0) | (color1->channels[B_V] >> 4);
}

/**
 * @brief Writes two 5-bit-per-channel colors to the compressed block using differential encoding.
 * @param block Pointer to the destination compressed block.
 * @param color0 The base color.
 * @param color1 The second color.
 */
void WriteColors555(__global uchar* block,
						   const Color* color0,
						   const Color* color1) {
	
	const uchar two_compl_trans_table[8] = {4, 5, 6, 7, 0, 1, 2, 3};
	
	short delta_r = (short)(color1->channels[R_V] >> 3) - (color0->channels[R_V] >> 3);
	short delta_g = (short)(color1->channels[G_V] >> 3) - (color0->channels[G_V] >> 3);
	short delta_b = (short)(color1->channels[B_V] >> 3) - (color0->channels[B_V] >> 3);
	
	block[0] = (color0->channels[R_V] & 0xf8) | two_compl_trans_table[delta_r + 4];
	block[1] = (color0->channels[G_V] & 0xf8) | two_compl_trans_table[delta_g + 4];
	block[2] = (color0->channels[B_V] & 0xf8) | two_compl_trans_table[delta_b + 4];
}

/**
 * @brief Writes the codeword table index for a sub-block.
 * @param block The compressed block.
 * @param sub_block_id The ID of the sub-block (0 or 1).
 * @param table The index of the codeword table to use.
 */
void WriteCodewordTable(__global uchar* block, uchar sub_block_id, uchar table) {
	uchar shift = (2 + (3 - sub_block_id * 3));
	block[3] &= ~(0x07 << shift);
	block[3] |= table << shift;
}

/**
 * @brief Writes the 32-bit pixel data (indices) to the compressed block.
 * @param block The compressed block.
 * @param pixel_data The 32 bits of pixel indices.
 */
void WritePixelData(__global uchar* block, uint pixel_data) {
	block[4] = (pixel_data >> 24) & 0xff;
	block[5] = (pixel_data >> 16) & 0xff;
	block[6] = (pixel_data >> 8) & 0xff;
	block[7] = pixel_data & 0xff;
}

// Writes the flip bit, which determines the sub-block split direction.
void WriteFlip(__global uchar* block, bool flip) {
	block[3] &= ~0x01;
	block[3] |= (uchar)flip;
}

// Writes the differential mode bit.
void WriteDiff(__global uchar* block, bool diff) {
	block[3] &= ~0x02;
	block[3] |= (uchar)(diff) << 1;
}

// Creates an expanded 8-bit color from 4-bit-per-channel components.
Color makeColor444(const float* bgr) {
	uchar b4 = round_to_4_bits(bgr[0]);
	uchar g4 = round_to_4_bits(bgr[1]);
	uchar r4 = round_to_4_bits(bgr[2]);
	Color bgr444;
	bgr444.channels[B_V] = (b4 << 4) | b4;
	bgr444.channels[G_V] = (g4 << 4) | g4;
	bgr444.channels[R_V] = (r4 << 4) | r4;
	bgr444.channels[A_V] = 0x44;
	return bgr444;
}

// Creates an expanded 8-bit color from 5-bit-per-channel components.
Color makeColor555(const float bgr[3]) {
	uchar b5 = round_to_5_bits(bgr[0]);
	uchar g5 = round_to_5_bits(bgr[1]);
	uchar r5 = round_to_5_bits(bgr[2]);
	Color bgr555;
	bgr555.channels[B_V] = (b5 << 3) | (b5 >> 2);
	bgr555.channels[G_V] = (g5 << 3) | (g5 >> 2);
	bgr555.channels[R_V] = (r5 << 3) | (r5 >> 2);
	bgr555.channels[A_V] = 0x55;
	return bgr555;
}

// Computes the average RGB color for a sub-block of 8 pixels.
void getAverageColor(const Color* src, float* avg_color)
{
	uint sum_b = 0, sum_g = 0, sum_r = 0;
	for (unsigned int i = 0; i < 8; ++i) {
		sum_b += src[i].channels[B_V];
		sum_g += src[i].channels[G_V];
		sum_r += src[i].channels[R_V];
	}
	const float kInv8 = 1.0f / 8.0f;
	avg_color[0] = (float)(sum_b) * kInv8;
	avg_color[1] = (float)(sum_g) * kInv8;
	avg_color[2] = (float)(sum_r) * kInv8;
}

/**
 * @brief Determines the best luminance modification table and pixel indices for a sub-block.
 * @param block The destination compressed block.
 * @param src The source pixels for the sub-block.
 * @param base The base color for the sub-block.
 * @param sub_block_id The ID of the sub-block.
 * @param idx_to_num_tab Lookup table for pixel index reordering.
 * @param threshold An error threshold for early exit.
 * @return The total error for the best encoding found.
 */
unsigned long computeLuminance(__global uchar* block,
						   const Color* src,
						   const Color* base,
						   int sub_block_id,
						   __constant const uchar* idx_to_num_tab,
						   unsigned long threshold)
{
	uint best_tbl_err = threshold;
	uchar best_tbl_idx = 0;
	uchar best_mod_idx[8][8];  

	// Iterate through all possible codeword tables.
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		// Generate candidate colors for the current table.
		Color candidate_color[4];  
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			candidate_color[mod_idx] = makeColor(base, lum);
		}
		
		uint tbl_err = 0;
		// For each pixel, find the best-matching candidate color.
		for (unsigned int i = 0; i < 8; ++i) {
			uint best_mod_err = threshold;
			for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
				const Color *color = &candidate_color[mod_idx];
				uint mod_err = getColorError(&src[i], color);
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
	
	// Pack the pixel indices into a 32-bit integer.
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
 * @param src The 16 source pixels.
 * @param error Pointer to store the resulting compression error.
 * @return 1 if the block was successfully compressed as a solid color, 0 otherwise.
 */
int tryCompressSolidBlock(__global uchar* dst,
						   const Color* src,
						   unsigned long* error)
{
	// Check if all pixels in the block are identical.
	for (unsigned int i = 1; i < 16; ++i) {
		if (src[i].bits != src[0].bits)
			return false;
	}
	
	for (int i = 0; i < 8; i++) dst[i] = 0; 

	float src_color_float[3] = {(float)(src->channels[B_V]),
		(float)(src->channels[G_V]),
		(float)(src->channels[R_V])};

	Color base = makeColor555(src_color_float);
	
	WriteDiff(dst, 1);
	WriteFlip(dst, 0);
	WriteColors555(dst, &base, &base);
	
	// Find the best luminance modification to represent the solid color.
	uchar best_tbl_idx = 0;
	uchar best_mod_idx = 0;
	uint best_mod_err = 0xffffffff; 
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			const Color color = makeColor(&base, lum);
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
	return 1;
}

/**
 * @brief Compresses a 4x4 pixel block.
 * @param dst Pointer to the 8-byte destination block.
 * @param ver_src Source pixels arranged for a vertical split.
 * @param hor_src Source pixels arranged for a horizontal split.
 * @param threshold Error threshold for early termination.
 * @return The total compression error for the block.
 *
 * This function is the core of the compression logic. It decides whether to
 * treat the block as a solid color, determines the best split (horizontal or
 * vertical), decides between differential or absolute color encoding, and
 * calls computeLuminance to find the best pixel indices.
 */
unsigned long compressBlock(__global uchar* dst,
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
	ushort use_differential[2] = {true, true};
	
	// Determine if differential or absolute color encoding should be used for
	// both potential splits (vertical and horizontal).
	for (unsigned int i = 0, j = 1; i < 4; i += 2, j += 2) {
		float avg_color_0[3];
		getAverageColor(sub_block_src[i], avg_color_0);
		Color avg_color_555_0 = makeColor555(avg_color_0);
		
		float avg_color_1[3];
		getAverageColor(sub_block_src[j], avg_color_1);
		Color avg_color_555_1 = makeColor555(avg_color_1);
		
		for (unsigned int light_idx = 0; light_idx < 3; ++light_idx) {
			int u = avg_color_555_0.channels[light_idx] >> 3;
			int v = avg_color_555_1.channels[light_idx] >> 3;
			
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
	
	// Calculate the error for both vertical and horizontal splits to decide
	// which one to use (the "flip" bit).
	uint sub_block_err[4] = {0};
	for (unsigned int i = 0; i < 4; ++i) {
		for (unsigned int j = 0; j < 8; ++j) {
			sub_block_err[i] += getColorError(&sub_block_avg[i], &sub_block_src[i][j]);
		}
	}
	
	bool flip = sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1];
	
	for (int i = 0; i < 8; i++) dst[i] = 0;

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
	
	// Compute luminance and pixel indices for the two chosen sub-blocks.
	lumi_error1 = computeLuminance(dst, sub_block_src[sub_block_off_0],
								   &sub_block_avg[sub_block_off_0], 0,
								   g_idx_to_num[sub_block_off_0],
								   threshold);
	
	lumi_error2 = computeLuminance(dst, sub_block_src[sub_block_off_1],
								   &sub_block_avg[sub_block_off_1], 1,
								   g_idx_to_num[sub_block_off_1],
								   threshold);
	
	return lumi_error1 + lumi_error2;
}

/**
 * @brief The main OpenCL kernel for texture compression.
 * @param width The width of the source image.
 * @param height The height of the source image.
 * @param src A global memory buffer containing the source image data.
 * @param dst A global memory buffer to store the compressed output.
 *
 * Each work-item in the kernel is responsible for compressing a single 4x4
 * block of pixels.
 */
__kernel void compress(const int width,
                const int height,
				__global uchar* src,
				__global uchar* dst)
{
	int y = get_global_id(0);
	int x = get_global_id(1);
    Color ver_blocks[16];
    Color hor_blocks[16];
	
	// Calculate offsets into the source and destination buffers.
	src += width * 4 * 4 * y;
    dst += (8 * width/4 * y + 8 * x);
    y = y*4;
    x = x*4;

	// Pointers to the rows of the 4x4 block.
	__global Color* row0 = (__global Color*)(src + x * 4);
	__global Color* row1 = row0 + width;
	__global Color* row2 = row1 + width;
	__global Color* row3 = row2 + width;

	// Read the 4x4 block and arrange pixels for both vertical and horizontal splits.
	memcpy_gpu(ver_blocks, row0, 8);
	memcpy_gpu(ver_blocks + 2, row1, 8);
	memcpy_gpu(ver_blocks + 4, row2, 8);
	memcpy_gpu(ver_blocks + 6, row3, 8);
	memcpy_gpu(ver_blocks + 8, row0 + 2, 8);
	memcpy_gpu(ver_blocks + 10, row1 + 2, 8);
	memcpy_gpu(ver_blocks + 12, row2 + 2, 8);
	memcpy_gpu(ver_blocks + 14, row3 + 2, 8);
	
	memcpy_gpu(hor_blocks, row0, 16);
	memcpy_gpu(hor_blocks + 4, row1, 16);
	memcpy_gpu(hor_blocks + 8, row2, 16);
	memcpy_gpu(hor_blocks + 12, row3, 16);
	
	// Compress the block.
	compressBlock(dst, ver_blocks, hor_blocks, 2147483647);
}