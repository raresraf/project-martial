/**
 * @file kernel.cl
 * @brief OpenCL implementation of the ETC1 (Ericsson Texture Compression) algorithm.
 * 
 * Functional Intent: Provides a parallelized implementation for compressing RGBA 
 * images into the ETC1 format. It processes 4x4 pixel blocks by determining 
 * the optimal sub-block partitioning (horizontal vs vertical), base color 
 * quantization (RGB444 vs RGB555), and codeword table selection to minimize 
 * perceptual error across high-dimensional vector data.
 * 
 * Domain: Graphics Programming, Image Compression (ETC1), Parallel Computing.
 */

#define INT32_MAX	2147483647
#define UINT32_MAX 	0xffffffff

typedef uchar	uint8_t;
typedef short 	int16_t;
typedef uint 	uint32_t;

typedef union u_Color {
	struct BgraColorType {
		uint8_t b;
		uint8_t g;
		uint8_t r;
		uint8_t a;
	} channels;
	uint8_t components[4];
	uint32_t bits;
} Color;

#define ALIGNAS(X)	__attribute__((aligned(X)))

void memcpy(void *destination, void *source, size_t num)
{
	char *c_destination = (char *) destination;
	char *c_source = (char *) source;

	for (int i = 0; i < num; i++)
		c_destination[i] = c_source[i];
}

void memset(__global void *ptr, int value, size_t num)
{
	__global char *c_ptr = (__global char *) ptr;
	while (num > 0) {
		*c_ptr = (unsigned char) value;
		c_ptr++;
		num--;
	}
}

/**
 * round_to_5_bits - Quantizes an 8-bit color component to 5 bits for differential mode.
 */
uint8_t round_to_5_bits(float val) {
	return (uint8_t) clamp(val * 31.0f / 255.0f + 0.5f, 0.0f, 31.0f);
}

/**
 * round_to_4_bits - Quantizes an 8-bit color component to 4 bits for individual mode.
 */
uint8_t round_to_4_bits(float val) {
	return (uint8_t) clamp(val * 15.0f / 255.0f + 0.5f, 0.0f, 15.0f);
}

// g_codeword_tables - Standard ETC1 luminance modifier tables.
ALIGNAS(16) __constant int16_t g_codeword_tables[8][4] = {
	{-8, -2, 2, 8},
	{-17, -5, 5, 17},
	{-29, -9, 9, 29},
	{-42, -13, 13, 42},
	{-60, -18, 18, 60},
	{-80, -24, 24, 80},
	{-106, -33, 33, 106},
	{-183, -47, 47, 183}};

// g_mod_to_pix - Maps codeword modifiers to ETC1 pixel index bits.
__constant uint8_t g_mod_to_pix[4] = {3, 2, 0, 1};

// g_idx_to_num - Maps sub-block local indices to global 4x4 block pixel ordinals.
__constant uint8_t g_idx_to_num[4][8] = {
	{0, 4, 1, 5, 2, 6, 3, 7},        
	{8, 12, 9, 13, 10, 14, 11, 15},  
	{0, 4, 8, 12, 1, 5, 9, 13},      
	{2, 6, 10, 14, 3, 7, 11, 15}     
};

/**
 * makeColor - Applies a luminance modifier to a base color and clamps to 8-bit range.
 */
Color makeColor(Color *base, int16_t lum) {
	int b = (int) (base->channels.b) + lum;
	int g = (int) (base->channels.g) + lum;
	int r = (int) (base->channels.r) + lum;

	Color color;
	color.channels.b = (uint8_t) (clamp(b, 0, 255));
	color.channels.g = (uint8_t) (clamp(g, 0, 255));
	color.channels.r = (uint8_t) (clamp(r, 0, 255));

	return color;
}

/**
 * getColorError - Computes the error between an original pixel and a compressed representation.
 * 
 * Algorithm: L2 (Euclidean) or Perceptual (Weighted) distance.
 */
uint32_t getColorError(Color *u, Color *v) {
#ifdef USE_PERCEIVED_ERROR_METRIC
	float delta_b = (float) (u->channels.b) - v->channels.b;
	float delta_g = (float) (u->channels.g) - v->channels.g;
	float delta_r = (float) (u->channels.r) - v->channels.r;
	return (uint32_t) (0.299f * delta_b * delta_b +
					   0.587f * delta_g * delta_g +
					   0.114f * delta_r * delta_r);
#else
	int delta_b = (int) (u->channels.b) - v->channels.b;
	int delta_g = (int) (u->channels.g) - v->channels.g;
	int delta_r = (int) (u->channels.r) - v->channels.r;
	return delta_b * delta_b + delta_g * delta_g + delta_r * delta_r;
#endif
}

/**
 * WriteColors444 - Packs two RGB444 base colors into the block header.
 */
void WriteColors444(__global uint8_t* block,
					Color *color0,
					Color *color1) {
	block[0] = (color0->channels.r & 0xf0) | (color1->channels.r >> 4);
	block[1] = (color0->channels.g & 0xf0) | (color1->channels.g >> 4);
	block[2] = (color0->channels.b & 0xf0) | (color1->channels.b >> 4);
}

/**
 * WriteColors555 - Packs RGB555 base color and its 3-bit differential into the block header.
 */
void WriteColors555(__global uint8_t* block,
					Color *color0,
					Color *color1) {
	uint8_t two_compl_trans_table[8] = {4, 5, 6, 7, 0, 1, 2, 3};
	
	int16_t delta_r = (int16_t) (color1->channels.r >> 3) - (color0->channels.r >> 3);
	int16_t delta_g = (int16_t) (color1->channels.g >> 3) - (color0->channels.g >> 3);
	int16_t delta_b = (int16_t) (color1->channels.b >> 3) - (color0->channels.b >> 3);
	
	block[0] = (color0->channels.r & 0xf8) | two_compl_trans_table[delta_r + 4];
	block[1] = (color0->channels.g & 0xf8) | two_compl_trans_table[delta_g + 4];
	block[2] = (color0->channels.b & 0xf8) | two_compl_trans_table[delta_b + 4];
}

/**
 * WriteCodewordTable - Sets the 3-bit table index for the sub-block.
 */
void WriteCodewordTable(__global uint8_t* block,
						uint8_t sub_block_id,
						uint8_t table) {
	uint8_t shift = (2 + (3 - sub_block_id * 3));
	block[3] &= ~(0x07 << shift);
	block[3] |= table << shift;
}

/**
 * WritePixelData - Packs the 32-bit modulation indices into the block.
 */
void WritePixelData(__global uint8_t* block, uint32_t pixel_data) {
	block[4] |= pixel_data >> 24;
	block[5] |= (pixel_data >> 16) & 0xff;
	block[6] |= (pixel_data >> 8) & 0xff;
	block[7] |= pixel_data & 0xff;
}

void WriteFlip(__global uint8_t* block, bool flip) {
	block[3] &= ~0x01;
	block[3] |= (uint8_t) (flip);
}

void WriteDiff(__global uint8_t* block, bool diff) {
	block[3] &= ~0x02;
	block[3] |= (uint8_t) (diff) << 1;
}

/**
 * makeColor444 - Reconstructs an 8-bit color from 4-bit components.
 */
Color makeColor444(float* bgr) {
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
 * makeColor555 - Reconstructs an 8-bit color from 5-bit components.
 */
Color makeColor555(float* bgr) {
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
 * getAverageColor - Computes the mean RGB color for a pixel set.
 */
void getAverageColor(Color* src, float* avg_color)
{
	uint32_t sum_b = 0, sum_g = 0, sum_r = 0;
	for (unsigned int i = 0; i < 8; ++i) {
		sum_b += src[i].channels.b;
		sum_g += src[i].channels.g;
		sum_r += src[i].channels.r;
	}
	float kInv8 = 1.0f / 8.0f;
	avg_color[0] = (float) (sum_b) * kInv8;
	avg_color[1] = (float) (sum_g) * kInv8;
	avg_color[2] = (float) (sum_r) * kInv8;
}

/**
 * computeLuminance - Heuristic search for the optimal codeword table.
 * 
 * Algorithm: Error-minimizing table selection.
 * Logic: Iterates through codeword tables and modulation indices to find 
 * the combination that best approximates the original pixel colors for a sub-block.
 */
unsigned long computeLuminance(__global uint8_t* block,
   							   Color* src,
							   Color* base,
							   int sub_block_id,
							   uint8_t sub_block_off,
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
				Color color = candidate_color[mod_idx];
				uint32_t mod_err = getColorError(src + i, &color);
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

	/**
	 * Block Logic: Modifier index serialization.
	 * Logic: Packs 2-bit modifier indices into the 32-bit pixel data field.
	 */
	uint32_t pix_data = 0;
	for (unsigned int i = 0; i < 8; ++i) {
		uint8_t mod_idx = best_mod_idx[best_tbl_idx][i];
		uint8_t pix_idx = g_mod_to_pix[mod_idx];
		uint32_t lsb = pix_idx & 0x1;
		uint32_t msb = pix_idx >> 1;
		int texel_num = g_idx_to_num[sub_block_off][i];
		pix_data |= msb << (texel_num + 16);
		pix_data |= lsb << (texel_num);
	}
	WritePixelData(block, pix_data);
	return best_tbl_err;
}


/**
 * tryCompressSolidBlock - Optimization for monochromatic blocks.
 */
bool tryCompressSolidBlock(__global uint8_t *dst,
						   Color *src,
						   unsigned long *error)
{
	for (unsigned int i = 1; i < 16; ++i) {
		if (src[i].bits != src[0].bits)
			return false;
	}
	
	memset(dst, 0, 8);
	float src_color_float[3] = {(float) (src->channels.b), (float) (src->channels.g), (float) (src->channels.r)};
	Color base = makeColor555(src_color_float);
	
	WriteDiff(dst, true);
	WriteFlip(dst, false);
	WriteColors555(dst, &base, &base);
	
	uint8_t best_tbl_idx = 0;
	uint8_t best_mod_idx = 0;
	uint32_t best_mod_err = UINT32_MAX;
	
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			int16_t lum = g_codeword_tables[tbl_idx][mod_idx];
			Color color = makeColor(&base, lum);
			uint32_t mod_err = getColorError(src, &color);
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
 * compressBlock - High-level decision logic for one 4x4 block.
 * 
 * Logic: Evaluates base color modes (differential vs individual) and 
 * sub-block partitioning (flip) to find the best perceptual fit.
 */
unsigned long compressBlock(__global uint8_t* dst,
							Color* ver_src,
							Color* hor_src,
							unsigned long threshold)
{
	unsigned long solid_error = 0;
	if (tryCompressSolidBlock(dst, ver_src, &solid_error)) {
		return solid_error;
	}
	
	Color *sub_block_src[4] = {ver_src, ver_src + 8, hor_src, hor_src + 8};
	Color sub_block_avg[4];
	bool use_differential[2] = {true, true};
	
	/**
	 * Block Logic: Mode Selection.
	 * Logic: Determines if the 3-bit differential color encoding fits the 
	 * sub-block deltas. Falls back to RGB444 if range is exceeded.
	 */
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
			sub_block_err[i] += getColorError(sub_block_avg + i, sub_block_src[i] + j);
		}
	}
	
	// Flip bit Logic: Determines if horizontal or vertical split is more efficient.
	bool flip = sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1];
	
	memset(dst, 0, 8);
	WriteDiff(dst, use_differential[!!flip]);
	WriteFlip(dst, flip);

	uint8_t sub_block_off_0 = flip ? 2 : 0;
	uint8_t sub_block_off_1 = sub_block_off_0 + 1;
	
	if (use_differential[!!flip]) {
		WriteColors555(dst, &sub_block_avg[sub_block_off_0], &sub_block_avg[sub_block_off_1]);
	} else {
		WriteColors444(dst, &sub_block_avg[sub_block_off_0], &sub_block_avg[sub_block_off_1]);
	}
	
	unsigned long lumi_error1 = 0, lumi_error2 = 0;
	lumi_error1 = computeLuminance(dst, sub_block_src[sub_block_off_0],
								   &sub_block_avg[sub_block_off_0], 0,
								   sub_block_off_0, threshold);
	
	lumi_error2 = computeLuminance(dst, sub_block_src[sub_block_off_1],
								   &sub_block_avg[sub_block_off_1], 1,
								   sub_block_off_1, threshold);
	
	return lumi_error1 + lumi_error2;
}

/**
 * @kernel compress
 * @brief Main parallel entry point for image compression.
 */
__kernel void compress(__global uchar *src,
					   __global uchar *dst,
					   int width,
					   int height)
{
	Color ver_blocks[16];
	Color hor_blocks[16];

	int y = get_global_id(0);
	int x = get_global_id(1);

	int offset_src = y * width * 4 * 4 + x * 4 * 4;
	int offset_dst = x * 8 + y * (width / 4) * 8;

	Color* row0 = src + offset_src;
	Color* row1 = row0 + width;
	Color* row2 = row1 + width;
	Color* row3 = row2 + width;
	
	// Block Logic: Memory gathering for sub-block candidates.
	memcpy(ver_blocks, row0, 8);
	memcpy(ver_blocks + 2, row1, 8);
	memcpy(ver_blocks + 4, row2, 8);
	memcpy(ver_blocks + 6, row3, 8);
	memcpy(ver_blocks + 8, row0 + 2, 8);
	memcpy(ver_blocks + 10, row1 + 2, 8);
	memcpy(ver_blocks + 12, row2 + 2, 8);
	memcpy(ver_blocks + 14, row3 + 2, 8);
	
	memcpy(hor_blocks, row0, 16);
	memcpy(hor_blocks + 4, row1, 16);
	memcpy(hor_blocks + 8, row2, 16);
	memcpy(hor_blocks + 12, row3, 16);
	
	compressBlock(dst + offset_dst, ver_blocks, hor_blocks, INT32_MAX);
}
