/**
 * @file compressor.cl
 * @brief OpenCL kernel and utility functions for parallel ETC1 texture compression.
 * 
 * Functional Intent: Implements the Ericsson Texture Compression (ETC1) 
 * algorithm for 4x4 pixel blocks. It orchestrates the transformation from 
 * BGRA8 input to the packed 64-bit ETC1 format by evaluating sub-block 
 * partitioning (horizontal vs vertical), base color quantization (individual 
 * vs differential modes), and optimal luminance codeword selection to 
 * minimize perceptual error.
 * 
 * Domain: Graphics Programming, Image Compression (ETC1), Parallel Computing.
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
 * fclamp - Floating-point clamping utility.
 */
inline float fclamp(float val, float min, float max) {
	return val < min ? min : (val > max ? max : val);
}

/**
 * iclamp - Integer clamping utility.
 */
inline int iclamp(int val, int min, int max) {
	return val < min ? min : (val > max ? max : val);
}

/**
 * myMemSet - Global memory zeroing utility for OpenCL.
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
 * round_to_5_bits - Quantizes an 8-bit color component to 5 bits for differential mode.
 */
inline uchar round_to_5_bits(float val) {
	return (uchar)fclamp(val * 31.0f / 255.0f + 0.5f, 0, 31);
}

/**
 * round_to_4_bits - Quantizes an 8-bit color component to 4 bits for individual mode.
 */
inline uchar round_to_4_bits(float val) {
	return (uchar)fclamp(val * 15.0f / 255.0f + 0.5f, 0, 15);
}

// g_codeword_tables - Standard ETC1 luminance modifier tables.
__constant short g_codeword_tables[8][4] __attribute__ ((aligned(16))) = {
	{-8, -2, 2, 8},
	{-17, -5, 5, 17},
	{-29, -9, 9, 29},
	{-42, -13, 13, 42},
	{-60, -18, 18, 60},
	{-80, -24, 24, 80},
	{-106, -33, 33, 106},
	{-183, -47, 47, 183}};

// g_mod_to_pix - Maps codeword modifiers to ETC1 pixel index bits.
__constant uchar g_mod_to_pix[4] = {3, 2, 0, 1};

// g_idx_to_num - Maps sub-block local indices to global 4x4 block pixel positions.
__constant uchar g_idx_to_num[4][8] = {
	{0, 4, 1, 5, 2, 6, 3, 7},       
	{8, 12, 9, 13, 10, 14, 11, 15},  
	{0, 4, 8, 12, 1, 5, 9, 13},      
	{2, 6, 10, 14, 3, 7, 11, 15}     
};

/**
 * makeColor - Applies a luminance modifier to a base color and clamps to 8-bit range.
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
 * getColorError - Computes the squared distance between two colors.
 */
inline uint getColorError(const Color u, const Color v) {
	int delta_b = (int)(u.channels.b) - v.channels.b;
	int delta_g = (int)(u.channels.g) - v.channels.g;
	int delta_r = (int)(u.channels.r) - v.channels.r;
	return delta_b * delta_b + delta_g * delta_g + delta_r * delta_r;
}

/**
 * WriteColors444 - Serializes two RGB444 base colors into the block header.
 */
inline void WriteColors444(__global uchar* block,
						   const Color color0,
						   const Color color1) {
	block[0] = (color0.channels.r & 0xf0) | (color1.channels.r >> 4);
	block[1] = (color0.channels.g & 0xf0) | (color1.channels.g >> 4);
	block[2] = (color0.channels.b & 0xf0) | (color1.channels.b >> 4);
}

/**
 * WriteColors555 - Serializes RGB555 base color and its 3-bit differential into the block header.
 */
inline void WriteColors555(__global uchar* block,
						   const Color color0,
						   const Color color1) {
	char two_compl_trans_table[8] = {4, 5, 6, 7, 0, 1, 2, 3};
	
	short delta_r = (short)(color1.channels.r >> 3) - (color0.channels.r >> 3);
	short delta_g = (short)(color1.channels.g >> 3) - (color0.channels.g >> 3);
	short delta_b = (short)(color1.channels.b >> 3) - (color0.channels.b >> 3);
	
	block[0] = (color0.channels.r & 0xf8) | two_compl_trans_table[delta_r + 4];
	block[1] = (color0.channels.g & 0xf8) | two_compl_trans_table[delta_g + 4];
	block[2] = (color0.channels.b & 0xf8) | two_compl_trans_table[delta_b + 4];
}

/**
 * WriteCodewordTable - Sets the 3-bit table index for the specific sub-block.
 */
inline void WriteCodewordTable(__global uchar* block,
							   uchar sub_block_id,
							   uchar table) {
	uchar shift = (2 + (3 - sub_block_id * 3));
	block[3] &= ~(0x07 << shift);
	block[3] |= table << shift;
}

/**
 * WritePixelData - Serializes the 32-bit modulation index map into the block.
 */
inline void WritePixelData(__global uchar* block, uint pixel_data) {
	block[4] |= pixel_data >> 24;
	block[5] |= (pixel_data >> 16) & 0xff;
	block[6] |= (pixel_data >> 8) & 0xff;
	block[7] |= pixel_data & 0xff;
}

inline void WriteFlip(__global uchar* block, bool flip) {
	block[3] &= ~0x01;
	block[3] |= (uchar)(flip);
}

inline void WriteDiff(__global uchar* block, bool diff) {
	block[3] &= ~0x02;
	block[3] |= (uchar)(diff) << 1;
}

/**
 * makeColor444 - Reconstructs a full 8-bit color from 4-bit components.
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
 * makeColor555 - Reconstructs a full 8-bit color from 5-bit components.
 */
inline Color makeColor555(const float* bgr) {
	uchar b5 = round_to_5_bits(bgr[0]);
	uchar g5 = round_to_5_bits(bgr[1]);
	uchar r5 = round_to_5_bits(bgr[2]);
	Color bgr555;
	bgr555.channels.b = (b5 > 2);
	bgr555.channels.g = (g5 > 2);
	bgr555.channels.r = (r5 > 2);
	bgr555.channels.a = 0x55;
	return bgr555;
}

/**
 * getAverageColor - Computes the mean RGB value for a sub-block.
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
 * computeLuminance - Exhaustive search for the optimal codeword table.
 * 
 * Algorithm: Error-minimizing table selection.
 * Logic: Iterates through all 8 codeword tables and selects the one that 
 * yields the minimum cumulative color error for the sub-block.
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

	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		Color candidate_color[4]; 
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			candidate_color[mod_idx] = makeColor(base, lum);
		}
		
		uint tbl_err = 0;
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
	 * Block Logic: Bitpacking.
	 * Logic: Packs optimal modifier indices into the 32-bit modulation map.
	 */
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
 * tryCompressSolidBlock - Optimization for monochromatic blocks.
 */
bool tryCompressSolidBlock(__global uchar* dst,
						   const Color* src,
						   unsigned long* error)
{
	for (unsigned int i = 1; i < 16; ++i) {
		if (src[i].bits != src[0].bits)
			return false;
	}
	
	myMemSet(dst, 0, 8);
	float src_color_float[3] = {(float)(src->channels.b), (float)(src->channels.g), (float)(src->channels.r)};
	Color base = makeColor555(src_color_float);
	
	WriteDiff(dst, true);
	WriteFlip(dst, false);
	WriteColors555(dst, base, base);
	
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
 * compressBlock - High-level decision engine for a single 4x4 block.
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
	bool use_differential[2] = {true, true};
	
	/**
	 * Block Logic: Mode Selection.
	 * Logic: Chooses between RGB444 and RGB555+3 modes based on the 
	 * component-wise delta between sub-blocks.
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
	
	uint sub_block_err[4] = {0};
	for (unsigned int i = 0; i < 4; ++i) {
		for (unsigned int j = 0; j < 8; ++j) {
			sub_block_err[i] += getColorError(sub_block_avg[i], sub_block_src[i][j]);
		}
	}
	
	// Decision Logic: Optimal partitioning (vertical vs horizontal).
	bool flip = sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1];

	myMemSet(dst, 0, 8);
	WriteDiff(dst, use_differential[!!flip]);
	WriteFlip(dst, flip);
	
	uchar sub_block_off_0 = flip ? 2 : 0;
	uchar sub_block_off_1 = sub_block_off_0 + 1;
	
	if (use_differential[!!flip]) {
		WriteColors555(dst, sub_block_avg[sub_block_off_0], sub_block_avg[sub_block_off_1]);
	} else {
		WriteColors444(dst, sub_block_avg[sub_block_off_0], sub_block_avg[sub_block_off_1]);
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


/**
 * copy_row - Extracts a row of texels from the global source buffer.
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
	}
}



void init_ver_block(Color *ver_blocks, Color * row0, Color * row1, Color * row2,
	Color * row3) {
	ver_blocks[0] = row0[0];
	ver_blocks[1] = row0[1];
	ver_blocks[2] = row1[0];
	ver_blocks[3] = row1[1];
	ver_blocks[4] = row2[0];
	ver_blocks[5] = row2[1];
	ver_blocks[6] = row3[0];
	ver_blocks[7] = row3[1];
	ver_blocks[8] = row0[2];
	ver_blocks[9] = row0[3];
	ver_blocks[10] = row1[2];
	ver_blocks[11] = row1[3];
	ver_blocks[12] = row2[2];
	ver_blocks[13] = row2[3];
	ver_blocks[14] = row3[2];
	ver_blocks[15] = row3[3];
}



void init_hor_block(Color * hor_blocks, Color * row0, Color * row1, Color * row2,
	Color * row3) {
	hor_blocks[0] = row0[0];
	hor_blocks[1] = row0[1];
	hor_blocks[2] = row0[2];
	hor_blocks[3] = row0[3];
	hor_blocks[4] = row1[0];
	hor_blocks[5] = row1[1];
	hor_blocks[6] = row1[2];
	hor_blocks[7] = row1[3];
	hor_blocks[8] = row2[0];
	hor_blocks[9] = row2[1];
	hor_blocks[10] = row2[2];
	hor_blocks[11] = row2[3];
	hor_blocks[12] = row3[0];
	hor_blocks[13] = row3[1];
	hor_blocks[14] = row3[2];
	hor_blocks[15] = row3[3];
}

/**
 * @kernel compressor
 * @brief Main parallel entry point for block-based texture compression.
 */
__kernel void
compressor(__global uchar *src,
        __global uchar *dst,
		int width,
        int height)
{
	Color ver_blocks[16];
	Color hor_blocks[16];
	
	int x = get_global_id(1)*4;

	// Block Logic: Source and destination offset calculation.
	src += get_global_id(0)*width*4*4 + x * 4;
	dst += get_global_id(0)*8*(width/4) + get_global_id(1)*8;

	Color row0[4], row1[4], row2[4], row3[4];
	copy_row(row0,src);
	copy_row(row1,src + width*4);
	copy_row(row2,src + 2*width*4);
	copy_row(row3,src + 3*width*4);

	init_ver_block(ver_blocks,row0,row1,row2,row3);
	init_hor_block(hor_blocks,row0,row1,row2,row3);

	compressBlock(dst, ver_blocks, hor_blocks, 2147483647);
}
