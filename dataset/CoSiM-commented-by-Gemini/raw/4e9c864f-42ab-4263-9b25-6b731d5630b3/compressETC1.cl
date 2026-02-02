/**
 * @file This file contains an OpenCL kernel for compressing textures using the ETC1 format.
 *
 * It defines the data structures and functions necessary to perform ETC1
 * compression on 4x4 blocks of pixels. The kernel includes color conversion
 * routines, error metric calculations, and the core compression logic that
 * determines the optimal encoding for each block.
 */

typedef unsigned int uint32_t;
typedef unsigned char uint8_t;
typedef short int16_t;

typedef union Color {
	struct BgraColorType {
		uint8_t b;
		uint8_t g;
		uint8_t r;
		uint8_t a;
	} channels;
	uint8_t components[4];
	uint32_t bits;
} Color;



void mymemcpyInt(__global uint8_t *dest, const uint8_t *src, int len)
{
    for(int i = 0; i < len; i++)
    	dest[i] = src[i];
}


void mymemcpy(Color *dest, __global Color *src, int len)
{
    for(int i = 0; i < len; i++)
    	dest[i] = src[i];
}


void mymemset(__global uint8_t *dest, int nr, int len)
{
    for(int i = 0; i < len; i++)
    	dest[i] = nr;
}

/**
 * @brief Clamps a value to a given range.
 *
 * @param val The value to clamp.
 * @param min The minimum value of the range.
 * @param max The maximum value of the range.
 * @return The clamped value.
 */
inline uint8_t compare(uint8_t val, uint8_t min, uint8_t max) {
	return (val < min ? min : (val > max ? max : val));
}

/**
 * @brief Clamps an integer value to a given range.
 *
 * @param val The value to clamp.
 * @param min The minimum value of the range.
 * @param max The maximum value of the range.
 * @return The clamped value.
 */
inline int compareInt(int val, int min, int max) {
	return (val < min ? min : (val > max ? max : val));
}
/**
 * @brief Rounds a float value to a 5-bit representation.
 * @param val The float value to round.
 * @return The 5-bit rounded value.
 */
inline uint8_t round_to_5_bits(float val) {
	return compare(val * 31.0f / 255.0f + 0.5f, 0, 31);
}
/**
 * @brief Rounds a float value to a 4-bit representation.
 * @param val The float value to round.
 * @return The 4-bit rounded value.
 */
inline uint8_t round_to_4_bits(float val) {
	return compare(val * 15.0f / 255.0f + 0.5f, 0, 15);
}


// ETC1 codeword tables for luminance modification.
__constant static int16_t g_codeword_tables[8][4] = {
	{-8, -2, 2, 8},
	{-17, -5, 5, 17},
	{-29, -9, 9, 29},
	{-42, -13, 13, 42},
	{-60, -18, 18, 60},
	{-80, -24, 24, 80},
	{-106, -33, 33, 106},
	{-183, -47, 47, 183}
};


// Maps a modifier index to a pixel index.
__constant static uint8_t g_mod_to_pix[4] = {3, 2, 0, 1};


// Maps a pixel index within a sub-block to its position in the final
// 32-bit pixel data.
__constant static uint8_t g_idx_to_num[4][8] = {
	{0, 4, 1, 5, 2, 6, 3, 7},        
	{8, 12, 9, 13, 10, 14, 11, 15},  


	{0, 4, 8, 12, 1, 5, 9, 13},      
	{2, 6, 10, 14, 3, 7, 11, 15}     
};

/**
 * @brief Creates a new color by applying a luminance offset to a base color.
 *
 * @param base The base color.
 * @param lum The luminance offset.
 * @return The new color.
 */
inline Color makeColor(const Color base, int16_t lum) {
	int b = (int)base.channels.b + lum;
	int g = (int)base.channels.g + lum;
	int r = (int)base.channels.r + lum;
	Color color;
	color.channels.b = (uint8_t)(compareInt(b, 0, 255));


	color.channels.g = (uint8_t)(compareInt(g, 0, 255));
	color.channels.r = (uint8_t)(compareInt(r, 0, 255));
	return color;
}


/**
 * @brief Calculates the squared error between two colors.
 *
 * @param u The first color.
 * @param v The second color.
 * @return The squared error.
 */
inline uint32_t getColorError(const Color u, const Color v) {
#ifdef USE_PERCEIVED_ERROR_METRIC
	float delta_b = (float)(u.channels.b) - v.channels.b;
	float delta_g = (float)(u.channels.g) - v.channels.g;
	float delta_r = (float)(u.channels.r) - v.channels.r;
	return (uint32_t)(0.299f * delta_b * delta_b +
						0.587f * delta_g * delta_g +
						0.114f * delta_r * delta_r);
#else
	int delta_b = (int)u.channels.b - v.channels.b;
	int delta_g = (int)u.channels.g - v.channels.g;
	int delta_r = (int)u.channels.r - v.channels.r;


	return delta_b * delta_b + delta_g * delta_g + delta_r * delta_r;
#endif
}
/**
 * @brief Writes two 4-bit colors to the ETC1 block.
 *
 * @param block A pointer to the ETC1 block.
 * @param color0 The first color.
 * @param color1 The second color.
 */
inline void WriteColors444(__global uint8_t* block,
						   const Color color0,
						   const Color color1) {
	
	block[0] = (color0.channels.r & 0xf0) | (color1.channels.r >> 4);
	block[1] = (color0.channels.g & 0xf0) | (color1.channels.g >> 4);


	block[2] = (color0.channels.b & 0xf0) | (color1.channels.b >> 4);
}
/**
 * @brief Writes two 5-bit colors with a 3-bit differential to the ETC1 block.
 *
 * @param block A pointer to the ETC1 block.
 * @param color0 The first color.
 * @param color1 The second color.
 */
inline void WriteColors555(__global uint8_t* block,
						   const Color color0,
						   const Color color1) {
	
	uint8_t two_compl_trans_table[8] = {
		4,  
		5,  
		6,  
		7,  
		0,  
		1,  
		2,  
		3,  
	};
	
	int16_t delta_r = (int16_t)(color1.channels.r >> 3) - (color0.channels.r >> 3);
	int16_t delta_g = (int16_t)(color1.channels.g >> 3) - (color0.channels.g >> 3);
	int16_t delta_b = (int16_t)(color1.channels.b >> 3) - (color0.channels.b >> 3);
	
	
	block[0] = (color0.channels.r & 0xf8) | two_compl_trans_table[delta_r + 4];
	block[1] = (color0.channels.g & 0xf8) | two_compl_trans_table[delta_g + 4];
	block[2] = (color0.channels.b & 0xf8) | two_compl_trans_table[delta_b + 4];
}
/**
 * @brief Writes the codeword table index for a sub-block to the ETC1 block.
 *
 * @param block A pointer to the ETC1 block.
 * @param sub_block_id The ID of the sub-block (0 or 1).
 * @param table The codeword table index.
 */
inline void WriteCodewordTable(__global uint8_t* block,
							   uint8_t sub_block_id,
							   uint8_t table) {
	
	uint8_t shift = (2 + (3 - sub_block_id * 3));
	block[3] &= ~(0x07 << shift);
	block[3] |= table << shift;
}
/**
 * @brief Writes the pixel data to the ETC1 block.
 *
 * @param block A pointer to the ETC1 block.
 * @param pixel_data The 32-bit pixel data.
 */
inline void WritePixelData(__global uint8_t* block, uint32_t pixel_data) {
	block[4] |= pixel_data >> 24;
	block[5] |= (pixel_data >> 16) & 0xff;
	block[6] |= (pixel_data >> 8) & 0xff;
	block[7] |= pixel_data & 0xff;
}
/**
 * @brief Writes the flip bit to the ETC1 block.
 *
 * @param block A pointer to the ETC1 block.
 * @param flip A boolean indicating whether to flip the block.
 */
inline void WriteFlip(__global uint8_t* block, bool flip) {
	block[3] &= ~0x01;
	block[3] |= (uint8_t)(flip);
}
/**
 * @brief Writes the differential bit to the ETC1 block.
 *
 * @param block A pointer to the ETC1 block.
 * @param diff A boolean indicating whether to use differential coding.
 */
inline void WriteDiff(__global uint8_t* block, bool diff) {
	block[3] &= ~0x02;
	block[3] |= (uint8_t)(diff) << 1;
}

/**
 * @brief Extracts a 4x4 block of pixels from a larger image.
 *
 * @param dst A pointer to the destination buffer for the 4x4 block.
 * @param src A pointer to the source image data.
 * @param width The width of the source image.
 */
inline void ExtractBlock(__global uint8_t* dst, const uint8_t* src, int width) {
	for (int j = 0; j < 4; ++j) {
		mymemcpyInt(&dst[j * 4 * 4], src, 4 * 4);
		src += width * 4;
	}
}




/**
 * @brief Converts a floating-point color to a 4:4:4 color.
 * @param bgr A pointer to an array of three floats representing the B, G, and R components.
 * @return The converted 4:4:4 color.
 */
inline Color makeColor444(const float* bgr) {
	uint8_t b4 = round_to_4_bits(bgr[0]);
	uint8_t g4 = round_to_4_bits(bgr[1]);
	uint8_t r4 = round_to_4_bits(bgr[2]);
	Color bgr444;
	bgr444.channels.b = (b4 << 4) | b4;
	bgr444.channels.g = (g4 << 4) | g4;
	bgr444.channels.r = (r4 << 4) | r4;
	
	bgr444.channels.a = 0x44;
	return bgr444;
}




/**
 * @brief Converts a floating-point color to a 5:5:5 color.
 * @param bgr A pointer to an array of three floats representing the B, G, and R components.
 * @return The converted 5:5:5 color.
 */
inline Color makeColor555(const float* bgr) {
	uint8_t b5 = round_to_5_bits(bgr[0]);
	uint8_t g5 = round_to_5_bits(bgr[1]);
	uint8_t r5 = round_to_5_bits(bgr[2]);
	Color bgr555;
	bgr555.channels.b = (b5 > 2);
	bgr555.channels.g = (g5 > 2);
	bgr555.channels.r = (r5 > 2);
	
	bgr555.channels.a = 0x55;
	return bgr555;
}
	
/**
 * @brief Calculates the average color of a sub-block.
 *
 * @param src A pointer to the source color data for the sub-block.
 * @param avg_color A pointer to an array of three floats to store the average color.
 */
void getAverageColor(const Color* src, float* avg_color)
{
	uint32_t sum_b = 0, sum_g = 0, sum_r = 0;
	
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
 * @brief Computes the luminance for a sub-block and determines the best
 *        codeword table and modifier indices.
 *
 * @param block A pointer to the ETC1 block.
 * @param src A pointer to the source color data for the sub-block.
 * @param base The base color of the sub-block.
 * @param sub_block_id The ID of the sub-block.
 * @param idx_to_num_tab A pointer to the index-to-number mapping table.
 * @param threshold The error threshold for early termination.
 * @return The total error for the sub-block.
 */
unsigned long computeLuminance(__global uint8_t* block,
						   const Color* src,
						   const Color base,
						   int sub_block_id,
						   __constant uint8_t* idx_to_num_tab,
						   unsigned long threshold)
{
	uint32_t best_tbl_err = threshold;
	uint8_t best_tbl_idx = 0;
	uint8_t best_mod_idx[8][8];  

	
	
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		
		
		Color candidate_color[4];  
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			int16_t lum = g_codeword_tables[tbl_idx][mod_idx];
			candidate_color[mod_idx] = makeColor(base, lum);
		}
		
		uint32_t tbl_err = 0;
		
		for (unsigned int i = 0; i < 8; ++i) {
			
			
			uint32_t best_mod_err = threshold;
			for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
				const Color color = candidate_color[mod_idx];
				
				uint32_t mod_err = getColorError(src[i], color);
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

	uint32_t pix_data = 0;

	for (unsigned int i = 0; i < 8; ++i) {
		uint8_t mod_idx = best_mod_idx[best_tbl_idx][i];
		uint8_t pix_idx = g_mod_to_pix[mod_idx];
		
		uint32_t lsb = pix_idx & 0x1;
		uint32_t msb = pix_idx >> 1;
		
		
		int texel_num = idx_to_num_tab[i];
		pix_data |= msb << (texel_num + 16);
		pix_data |= lsb << (texel_num);
	}

	WritePixelData(block, pix_data);

	return best_tbl_err;
}

/**
 * @brief Attempts to compress a block as a solid color.
 *
 * @param dst A pointer to the destination ETC1 block.
 * @param src A pointer to the source 4x4 block of pixels.
 * @param error A pointer to store the compression error.
 * @return True if the block was successfully compressed as a solid color,
 *         false otherwise.
 */
bool tryCompressSolidBlock(__global uint8_t* dst,
						   const Color* src,
						   unsigned long* error)
{
	for (unsigned int i = 1; i < 16; ++i) {
		if (src[i].bits != src[0].bits)
			return false;
	}
	
	
	mymemset(dst, 0, 8);
	
	float src_color_float[3] = {(float)(src->channels.b),
		(float)(src->channels.g),
		(float)(src->channels.r)};
	Color base = makeColor555(src_color_float);
	


	WriteDiff(dst, true);
	WriteFlip(dst, false);
	WriteColors555(dst, base, base);
	
	uint8_t best_tbl_idx = 0;
	uint8_t best_mod_idx = 0;
	uint32_t best_mod_err = 0xffffffff; 
	
	
	
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		
		
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			int16_t lum = g_codeword_tables[tbl_idx][mod_idx];
			const Color color = makeColor(base, lum);
			
			uint32_t mod_err = getColorError(*src, color);
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
	
	uint8_t pix_idx = g_mod_to_pix[best_mod_idx];
	uint32_t lsb = pix_idx & 0x1;
	uint32_t msb = pix_idx >> 1;
	
	uint32_t pix_data = 0;
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
 * @brief Compresses a 4x4 block of pixels into the ETC1 format.
 *
 * @param dst A pointer to the destination ETC1 block.
 * @param ver_src A pointer to the source pixels arranged for vertical split.
 * @param hor_src A pointer to the source pixels arranged for horizontal split.
 * @param threshold The error threshold for early termination.
 * @return The total compression error.
 */
unsigned long compressBlock(__global uint8_t* dst,
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
	
	
	
	
	uint32_t sub_block_err[4] = {0};
	for (unsigned int i = 0; i < 4; ++i) {
		for (unsigned int j = 0; j < 8; ++j) {
			sub_block_err[i] += getColorError(sub_block_avg[i], sub_block_src[i][j]);
		}
	}
	
	bool flip =
	sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1];
	
	
	mymemset(dst, 0, 8);
	
	WriteDiff(dst, use_differential[!!flip]);
	WriteFlip(dst, flip);
	
	uint8_t sub_block_off_0 = flip ? 2 : 0;
	uint8_t sub_block_off_1 = sub_block_off_0 + 1;
	
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


__kernel void
compressFunc(__global uint8_t* src,
	__global uint8_t* dst,
	int width,
	int height)
{

	unsigned long compresserror; 
	Color ver_blocks[16];
	Color hor_blocks[16];

	int y = get_global_id(0);	
	int x = get_global_id(1);	

	
	__global Color* row0 = (__global Color*)(src + 16 *(y * width + x));
	__global Color* row1 = row0 + width;
	__global Color* row2 = row1 + width;
	__global Color* row3 = row2 + width;

	mymemcpy(ver_blocks, row0, 8);
	mymemcpy(ver_blocks + 2, row1, 8);
	mymemcpy(ver_blocks + 4, row2, 8);
	mymemcpy(ver_blocks + 6, row3, 8);
	mymemcpy(ver_blocks + 8, row0 + 2, 8);
	mymemcpy(ver_blocks + 10, row1 + 2, 8);
	mymemcpy(ver_blocks + 12, row2 + 2, 8);
	mymemcpy(ver_blocks + 14, row3 + 2, 8);
			
	mymemcpy(hor_blocks, row0, 16);
	mymemcpy(hor_blocks + 4, row1, 16);
	mymemcpy(hor_blocks + 8, row2, 16);
	mymemcpy(hor_blocks + 12, row3, 16);

	compresserror = compressBlock(dst + 8 * (width / 4 * y + x), ver_blocks, hor_blocks, 0xffffffff);
	

}