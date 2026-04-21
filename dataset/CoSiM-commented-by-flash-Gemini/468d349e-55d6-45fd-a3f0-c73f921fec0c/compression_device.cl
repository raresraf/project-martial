/**
 * @file compression_device.cl
 * @brief OpenCL kernel and utility functions for ETC1 texture compression.
 * 
 * Functional Intent: Implements the Ericsson Texture Compression (ETC1) 
 * algorithm for 4x4 pixel blocks. It handles the transformation from BGRA8 
 * input to the packed 64-bit ETC1 format, including sub-block partitioning 
 * (horizontal vs vertical), base color quantization (RGB444/555), and 
 * optimal luminance codeword table selection to minimize perceptual error.
 * 
 * Domain: Graphics Programming, Image Compression (ETC1), Parallel Computing (OpenCL).
 */

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

/**
 * my_clamp - Scalar clamping utility.
 */
inline uint my_clamp(int val, int min, int max) {
	if (val < min)
		return min;
	else if (val > max)
		return max;
	return val;
}

/**
 * round_to_5_bits - Quantizes an 8-bit color channel to 5 bits for differential mode.
 */
inline uchar round_to_5_bits(int val) {
	return (uchar) my_clamp(val * 31.0f / 255.0f + 0.5f, 0, 31);
}

/**
 * round_to_4_bits - Quantizes an 8-bit color channel to 4 bits for individual mode.
 */
inline uchar round_to_4_bits(int val) {
	return (uchar) my_clamp(val * 15.0f / 255.0f + 0.5f, 0, 15);
}

// g_mod_to_pix - Maps codeword modifiers to ETC1 pixel index bits.
__constant short g_mod_to_pix[4] = {3, 2, 0, 1};

/**
 * makeColor - Applies a luminance modifier to a base color.
 */
inline union Color* makeColor(union Color base, short lum) {
	int b = (int)base.channels.b + (int)lum;
	int g = (int)base.channels.g + (int)lum;
	int r = (int)base.channels.r + (int)lum;
	union Color* color;
	color->channels.b = (uchar)(clamp(b, 0, 255));
	color->channels.g = (uchar)(clamp(g, 0, 255));
	color->channels.r = (uchar)(clamp(r, 0, 255));
	return (union Color*) color;
}

/**
 * getColorError - Computes the squared distance between two colors.
 * 
 * Algorithm: L2 Distance (Euclidean) or Perceptual Weighted Error.
 * Logic: Minimizes the error between the original pixel and the compressed 
 * representation (base + codeword modifier).
 */
inline uint getColorError(union Color u, union Color v) {
#ifdef USE_PERCEIVED_ERROR_METRIC
	float delta_b = (float)(u.channels.b) - v.channels.b;
	float delta_g = (float)(u.channels.g) - v.channels.g;
	float delta_r = (float)(u.channels.r) - v.channels.r;
	return (uint)(0.299f * delta_b * delta_b +
								 0.587f * delta_g * delta_g +
								 0.114f * delta_r * delta_r);
#else
	int delta_b = (int)(u.channels.b) - v.channels.b;
	int delta_g = (int)(u.channels.g) - v.channels.g;
	int delta_r = (int)(u.channels.r) - v.channels.r;
	return delta_b * delta_b + delta_g * delta_g + delta_r * delta_r;
#endif
}

/**
 * WriteColors444 - Serializes two RGB444 base colors into the block header.
 */
inline void WriteColors444(__global uchar* block,
						    union Color color0,
						    union Color color1
								) {
	block[0] = (color0.channels.r & 0xf0) | (color1.channels.r >> 4);
	block[1] = (color0.channels.g & 0xf0) | (color1.channels.g >> 4);
	block[2] = (color0.channels.b & 0xf0) | (color1.channels.b >> 4);
}

/**
 * WriteColors555 - Serializes RGB555 base color and its 3-bit differential into the block header.
 */
inline void WriteColors555(__global uchar* block,
						   union Color color0,
						   union Color color1) {
	uchar two_compl_trans_table[8] = {4, 5, 6, 7, 0, 1, 2, 3};

	short delta_r = (short)(color1.channels.r >> 3) - (color0.channels.r >> 3);
	short delta_g = (short)(color1.channels.g >> 3) - (color0.channels.g >> 3);
	short delta_b = (short)(color1.channels.b >> 3) - (color0.channels.b >> 3);

	block[0] = (color0.channels.r & 0xf8) | two_compl_trans_table[delta_r + 4];
	block[1] = (color0.channels.g & 0xf8) | two_compl_trans_table[delta_g + 4];
	block[2] = (color0.channels.b & 0xf8) | two_compl_trans_table[delta_b + 4];
}

/**
 * WriteCodewordTable - Sets the 3-bit table index for a specific sub-block.
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

inline void memcpy(uchar *dst, uchar *src, int width) {
	for (int i = 0; i < width; i++) {
		dst[i] = src[i];
	}
}

/**
 * makeColor444 - Reconstructs a full 8-bit color from 4-bit components.
 */
inline union Color makeColor444(float* bgr) {
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

/**
 * makeColor555 - Reconstructs a full 8-bit color from 5-bit components.
 */
inline union Color makeColor555(float* bgr) {
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

/**
 * getAverageColor - Computes the mean RGB value for a sub-block.
 */
void getAverageColor(union Color* src, float* avg_color)
{
	uint sum_b = 0, sum_g = 0, sum_r = 0;
	for (uint i = 0; i < 8; ++i) {
		sum_b += src[i].channels.b;
		sum_g += src[i].channels.g;
		sum_r += src[i].channels.r;
	}
	float kInv8 = 1.0f / 8.0f;
	avg_color[0] = (float)(sum_b) * kInv8;
	avg_color[1] = (float)(sum_g) * kInv8;
	avg_color[2] = (float)(sum_r) * kInv8;
}

void memset(__global uchar* dst, int value, int size) {
	for (int i = 0; i < size; i++) {
		dst[i] = value;
	}
}

/**
 * computeLuminance - Exhaustive search for the best luminance codeword table.
 * 
 * Algorithm: Per-subblock error minimization.
 * 1. Iterates through all 8 standard ETC1 codeword tables.
 * 2. For each table, finds the modifier index that minimizes color error for each pixel.
 * 3. Selects the table with the lowest cumulative error for the subblock.
 */
unsigned long computeLuminance(__global uchar* block,
						   union Color* src,
						   union Color base,
						   int sub_block_id,
						   uchar* idx_to_num_tab,
						   unsigned long threshold)
{
	short g_codeword_tables[8][4] = {
	{-8, -2, 2, 8},
	{-17, -5, 5, 17},
	{-29, -9, 9, 29},
	{-42, -13, 13, 42},
	{-60, -18, 18, 60},
	{-80, -24, 24, 80},
	{-106, -33, 33, 106},
	{-183, -47, 47, 183}};

	uint best_tbl_err = threshold;
	uchar best_tbl_idx = 0;
	uchar best_mod_idx[8][8];  

	// Block Logic: Table search loop.
	for (uint tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		union Color candidate_color[4];  
		for (uint mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			candidate_color[mod_idx] = *makeColor(base, lum);
		}

		uint tbl_err = 0;
		for (uint i = 0; i < 8; ++i) {
			uint best_mod_err = threshold;
			for (uint mod_idx = 0; mod_idx < 4; ++mod_idx) {
				union Color color = candidate_color[mod_idx];
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
	 * Block Logic: Bitstream packing.
	 * Logic: Converts optimal modifier indices into the split MSB/LSB bitplanes 
	 * required by the ETC1 format.
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
 * tryCompressSolidBlock - Fast-path optimization for monochrome blocks.
 */
bool tryCompressSolidBlock(__global uchar* dst,
						   union Color* src,
						   unsigned long* error)
{
	short g_codeword_tables[8][4] = {
	{-8, -2, 2, 8},
	{-17, -5, 5, 17},
	{-29, -9, 9, 29},
	{-42, -13, 13, 42},
	{-60, -18, 18, 60},
	{-80, -24, 24, 80},
	{-106, -33, 33, 106},
	{-183, -47, 47, 183}};

	uchar g_idx_to_num[4][8] = {
	{0, 4, 1, 5, 2, 6, 3, 7},        
	{8, 12, 9, 13, 10, 14, 11, 15},  
	{0, 4, 8, 12, 1, 5, 9, 13},      
	{2, 6, 10, 14, 3, 7, 11, 15}     
	};

	// Pre-condition: Block must have identical color in all 16 texels.
	for (unsigned int i = 1; i < 16; ++i) {
		if (src[i].bits != src[0].bits)
			return false;
	}

	memset(dst, 0, 8);
	float src_color_float[3] = {(float)(src->channels.b), (float)(src->channels.g), (float)(src->channels.r)};
	union Color base = makeColor555(src_color_float);

	WriteDiff(dst, true);
	WriteFlip(dst, false);
	WriteColors555(dst, base, base);

	uchar best_tbl_idx = 0;
	uchar best_mod_idx = 0;
	uint best_mod_err = UINT_MAX;

	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			union Color* color = makeColor(base, lum);
			uint mod_err = getColorError(*src, *color);
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
 * 
 * Logic: Chooses between differential (RGB555+3) and individual (RGB444) 
 * color modes, and evaluates the best subblock split (vertical vs horizontal).
 */
ulong compressBlock(__global uchar* dst,
										union Color* ver_src,
										union Color* hor_src,
										ulong threshold) {

	unsigned long solid_error = 0;
	if (tryCompressSolidBlock(dst, ver_src, &solid_error)) {
		return solid_error;
	}

	union Color* sub_block_src[4] = {ver_src, ver_src + 8, hor_src, hor_src + 8};
	union Color sub_block_avg[4];
	bool use_differential[2] = {true, true};

	/**
	 * Block Logic: Color Mode selection.
	 * Logic: Determines if the component-wise delta between sub-blocks 
	 * fits within the signed 3-bit range [-4, 3]. If not, falls back to 
	 * the less precise RGB444 mode.
	 */
	for (uint i = 0, j = 1; i < 4; i += 2, j += 2) {
		float avg_color_0[3];
		getAverageColor(sub_block_src[i], avg_color_0);
		union Color avg_color_555_0 = makeColor555(avg_color_0);

		float avg_color_1[3];
		getAverageColor(sub_block_src[j], avg_color_1);
		union Color avg_color_555_1 = makeColor555(avg_color_1);

		for (uint light_idx = 0; light_idx < 3; ++light_idx) {
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
	for (uint i = 0; i < 4; ++i) {
		for (uint j = 0; j < 8; ++j) {
			sub_block_err[i] += getColorError(sub_block_avg[i], sub_block_src[i][j]);
		}
	}

	// Decision Logic: Optimal partitioning (Flip bit).
	bool flip = sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1];

	memset(dst, 0, 8);
	WriteDiff(dst, use_differential[!!flip]);
	WriteFlip(dst, flip);

	uchar sub_block_off_0 = flip ? 2 : 0;
	uchar sub_block_off_1 = sub_block_off_0 + 1;

	if (use_differential[!!flip]) {
		WriteColors555(dst, sub_block_avg[sub_block_off_0], sub_block_avg[sub_block_off_1]);
	} else {
		WriteColors444(dst, sub_block_avg[sub_block_off_0], sub_block_avg[sub_block_off_1]);
	}

	return 0;
}

/**
 * memcpy_colors - Extracts color texels from the global source image.
 */
inline void memcpy_colors(union Color* blocks, int index,
			int offset, __global uchar *src)
{
		uchar *values1 = (uchar *) &blocks[index];
		uchar *values2 = (uchar *) &blocks[index + 1];
		for (int i = 0; i < 4; i++) {
				values1[i] = *(src+ offset + i);
				values2[i] = *(src+ offset + 4 + i);
		}
}


/**
 * @kernel compression_kernel
 * @brief Entry point for parallel block compression.
 * 
 * Logic: Dispatches a work-item for every 4x4 block in the input image. 
 * Reconstructs vertical and horizontal subblock candidates and delegates 
 * the algorithmic decision-making to `compressBlock`.
 */
__kernel void
compression_kernel(__global uchar* src,
		__global uchar* dst,
		int width,
    int height)
{
	int gid_0 = get_global_id(0);
	int gid_1 = get_global_id(1);

	union Color ver_blocks[16];
	union Color hor_blocks[16];

	int src_offset = 4 * 4 * gid_0 + gid_1 * width * 4 * 4;
	int dst_offset = gid_0 * 8 + 8 * gid_1 * width / 4;

	src += src_offset;

	// Block Logic: Sub-block extraction.
	for (int x = 0; x < width; x += 4) {
		memcpy_colors(ver_blocks, 0, 0, src + x);
	 	memcpy_colors(ver_blocks, 2, width, src + x);
		memcpy_colors(ver_blocks, 4, width * 2, src + x);
		memcpy_colors(ver_blocks, 6, width * 3, src + x);

		memcpy_colors(hor_blocks, 0, 0, src + x);
		memcpy_colors(hor_blocks, 2, 0, src + x + 8);
		memcpy_colors(hor_blocks, 4, width, src + x);
		memcpy_colors(hor_blocks, 6, width, src + x + 8);

		memcpy_colors(ver_blocks, 8, 0, src + x + 8);
	 	memcpy_colors(ver_blocks, 10, width, src + x + 8);
		memcpy_colors(ver_blocks, 12, width * 2, src + x + 8);
		memcpy_colors(ver_blocks, 14, width * 3, src + x + 8);

		memcpy_colors(hor_blocks, 8, 2 * width, src + x);
	 	memcpy_colors(hor_blocks, 10, 2 * width, src + x + 8);
		memcpy_colors(hor_blocks, 12, 3 * width, src + x);
		memcpy_colors(hor_blocks, 14, 3 * width, src + x + 8);
    }
	
    dst += dst_offset;
	compressBlock(dst, ver_blocks, hor_blocks, UINT_MAX);
}
