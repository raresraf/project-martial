/**
 * @file compress_device.cl
 * @brief OpenCL kernel logic for parallel ETC1 texture compression.
 * 
 * Architectural Intent: Implements the Ericsson Texture Compression (ETC1) algorithm
 * optimized for GPU parallel execution. The core logic involves processing 4x4 texel blocks
 * to find optimal base colors and luminance modifiers that minimize perceptual error.
 * 
 * Domain-Awareness:
 * - Uses a massively parallel NDRange grid where each work-item processes a 4x4 block.
 * - Exploits memory hierarchy by caching block data in private registers (`ver_blocks`, `hor_blocks`).
 * - Implements differential color encoding (555 mode) and standard (444 mode) based on component deltas.
 */

/**
 * @struct Color
 * @brief Represents a single color point in BGRA format with bitfield overlay.
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
 * @brief Clamps an integer value to the uchar range [min, max].
 */
uchar wrapper_clamp(int val, uchar min, uchar max){
	return (uchar)(val < min ? min : (val > max ? max : val));
}

/**
 * @brief Clamps a uchar value to the range [min, max].
 */
uchar wrapper_clamp2(uchar val, uchar min, uchar max){
	return (uchar)(val < min ? min : (val > max ? max : val));
}

/**
 * @brief Quantizes a float color component to 5 bits.
 */
inline uchar round_to_5_bits(float val) {
	return wrapper_clamp2((uchar)(val * 31.0f / 255.0f + 0.5f), (uchar)0, (uchar)31);
}

/**
 * @brief Quantizes a float color component to 4 bits.
 */
inline uchar round_to_4_bits(float val) {
	return wrapper_clamp2((uchar)(val * 15.0f / 255.0f + 0.5f),(uchar) 0,(uchar) 15);
}

/**
 * @brief Adjusts a base color by a luminance offset.
 * Functional Utility: Computes a candidate color for the codeword table evaluation.
 */
inline union Color makeColor(union Color base, short lum) {
	int b = (int)((int)(base.channels.b) + lum);
	int g = (int)((int)(base.channels.g) + lum);
	int r = (int)((int)(base.channels.r) + lum);
	union Color color;
	color.channels.b = (uchar)(wrapper_clamp(b, 0, 255));
	color.channels.g = (uchar)(wrapper_clamp(g, 0, 255));
	color.channels.r = (uchar)(wrapper_clamp(r, 0, 255));
	return color;
}

/**
 * @brief Bit-packs two colors into the 444 (non-differential) block format.
 */
inline void WriteColors444(__global uchar* block,
						   union Color color0,
						   union Color color1) {
	
	block[0] = (color0.channels.r & 0xf0) | (color1.channels.r >> 4);
	block[1] = (color0.channels.g & 0xf0) | (color1.channels.g >> 4);
	block[2] = (color0.channels.b & 0xf0) | (color1.channels.b >> 4);
}

/**
 * @brief Bit-packs colors into the 555 (differential) block format.
 * Algorithm: Computes 3-bit deltas between sub-blocks and applies two's complement transformation.
 */
inline void WriteColors555(__global uchar* block,
						   union Color color0,
						   union Color color1) {
	
	const uchar two_compl_trans_table[8] = {
		4,  
		5,  
		6,  
		7,  
		0,  
		1,  
		2,  
		3,  
	};

	short delta_r =	(short)((color1.channels.r >> 3) - (color0.channels.r >> 3));
	short delta_g = (short) ((color1.channels.g >> 3) - (color0.channels.g >> 3));
	short delta_b = (short) ((color1.channels.b >> 3) - (color0.channels.b >> 3));

	
	block[0] = (color0.channels.r & 0xf8) | two_compl_trans_table[delta_r + 4];
	block[1] = (color0.channels.g & 0xf8) | two_compl_trans_table[delta_g + 4];
	block[2] = (color0.channels.b & 0xf8) | two_compl_trans_table[delta_b + 4];
}

/**
 * @brief Encodes the codeword table index for a sub-block.
 */
inline void WriteCodewordTable(__global uchar* block,
							   uchar sub_block_id,
							   uchar table) {

	uchar shift = (2 + (3 - sub_block_id * 3));
	block[3] &= ~(0x07 << shift);
	block[3] |= table << shift;
}

/**
 * @brief Encodes pixel-level luminance modifiers.
 */
inline void WritePixelData(__global uchar* block, uint pixel_data) {
	block[4] |= pixel_data >> 24;
	block[5] |= (pixel_data >> 16) & 0xff;
	block[6] |= (pixel_data >> 8) & 0xff;
	block[7] |= pixel_data & 0xff;
}

/**
 * @brief Sets the block orientation flag (Flip bit).
 */
inline void WriteFlip(__global uchar* block, bool flip) {
	block[3] &= ~0x01;
	block[3] |= (uchar)(flip);
}

/**
 * @brief Sets the color encoding mode (Differential bit).
 */
inline void WriteDiff(__global uchar* block, bool diff) {
	block[3] &= ~0x02;
	block[3] |= (uchar)(diff) << 1;
}

/**
 * @brief Extracts a 4x4 texel block from the source image.
 * Functional Utility: Handles row-major to block-local memory mapping.
 */
inline void ExtractBlock(uchar* dst, const uchar* src, int width) {
	/**
	 * Block Logic: Row-wise block extraction.
	 * Invariant: Moves src pointer by row-width to fetch successive scanlines.
	 */
	for (int j = 0; j < 4; ++j) {
		
		for (int k = 0; k < 16; k++) {
			dst[j * 16 + k ] = *(src + k);
		}


		src += width * 4;
	}
}

/**
 * @brief Quantizes float color to 4-bit representation.
 */
inline union Color makeColor444(const float* bgr) {
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
 * @brief Quantizes float color to 5-bit representation.
 */
inline union Color makeColor555(const float* bgr) {
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
 * @brief Computes perceptual or Euclidean error between two colors.
 */
inline uint getColorError(union Color u, union Color v) {
#ifdef USE_PERCEIVED_ERROR_METRIC
	float delta_b = (float)((u.channels.b) - v.channels.b);
	float delta_g = (float)((u.channels.g) - v.channels.g);
	float delta_r = (float)((u.channels.r) - v.channels.r);
	// Functional Utility: Applies perceptual weighting constants to component errors.
	return (uint)(0.299f * delta_b * delta_b +
								 0.587f * delta_g * delta_g +
								 0.114f * delta_r * delta_r);
#else
	int delta_b = (int)((u.channels.b) - v.channels.b);
	int delta_g = (int)((u.channels.g) - v.channels.g);
	int delta_r = (int)((u.channels.r) - v.channels.r);
	return delta_b * delta_b + delta_g * delta_g + delta_r * delta_r;
#endif
}

/**
 * @brief Calculates the average color for a 2x4 sub-block.
 */
void getAverageColor(const union Color* src, float* avg_color)
{
	uint sum_b = 0, sum_g = 0, sum_r = 0;

	/**
	 * Block Logic: Energy accumulation.
	 * Invariant: Sums components over the 8 texels of the sub-block.
	 */
	for (unsigned int i = 0; i < 8; ++i) {
		sum_b += src[i].channels.b;
		sum_g += src[i].channels.g;
		sum_r += src[i].channels.r;
	}

	const float kInv8 = 1.0f / 8.0f;
	avg_color[0] = (float)((sum_b) * kInv8);


	avg_color[1] = (float)((sum_g) * kInv8);
	avg_color[2] = (float)((sum_r) * kInv8);
}

/**
 * @brief Optimizes luminance modifiers for a sub-block to minimize perceptual error.
 * Algorithm: Brute-force search over all 8 codeword tables and their modifiers.
 */
unsigned long computeLuminance(__global uchar* block,
						   const union Color* src,


						   union Color base,
						   int sub_block_id,
						   const uchar* idx_to_num_tab,
						   unsigned long threshold)
{
	uint best_tbl_err = threshold;
	uchar best_tbl_idx = 0;
	uchar best_mod_idx[8][8];  

	const short g_codeword_tables[8][4] __attribute__((aligned(16))) = {
		{-8, -2, 2, 8},
		{-17, -5, 5, 17},
		{-29, -9, 9, 29},
		{-42, -13, 13, 42},
		{-60, -18, 18, 60},
		{-80, -24, 24, 80},
		{-106, -33, 33, 106},
		{-183, -47, 47, 183}};



	const uchar g_mod_to_pix[4] = {3, 2, 0, 1};



	
	
	/**
	 * Block Logic: Codeword table evaluation.
	 * Invariant: Iterates through all 8 standard tables to find the best luminance range.
	 */
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		
		
		union Color candidate_color[4];  
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			candidate_color[mod_idx] = makeColor(base, lum);
		}

		uint tbl_err = 0;

		/**
		 * Block Logic: Texel-level modifier selection.
		 * Invariant: Selects the modifier (0-3) that minimizes error for the current texel.
		 */
		for (unsigned int i = 0; i < 8; ++i) {
			
			
			uint best_mod_err = threshold;
			for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
				union Color color = candidate_color[mod_idx];

				uint mod_err = getColorError(src[i], color);
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

	/**
	 * Block Logic: Pixel data bitstream generation.
	 * Invariant: Packs 2-bit modifier indices into the 32-bit payload.
	 */
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
 * @brief Optimized path for homogeneous color blocks.
 */
bool tryCompressSolidBlock(__global uchar* dst,
						   const union Color* src,
						   unsigned long* error)
{
	const uchar g_mod_to_pix[4] = {3, 2, 0, 1};

	const short g_codeword_tables[8][4] __attribute__((aligned(16))) = {
	{-8, -2, 2, 8},
	{-17, -5, 5, 17},
	{-29, -9, 9, 29},
	{-42, -13, 13, 42},
	{-60, -18, 18, 60},
	{-80, -24, 24, 80},
	{-106, -33, 33, 106},
	{-183, -47, 47, 183}};

	const uchar g_idx_to_num[4][8] = {
	{0, 4, 1, 5, 2, 6, 3, 7},        
	{8, 12, 9, 13, 10, 14, 11, 15},  
	{0, 4, 8, 12, 1, 5, 9, 13},      
	{2, 6, 10, 14, 3, 7, 11, 15}     
};

	/**
	 * Block Logic: Uniformity check.
	 * Invariant: Short-circuits if all bits match the first texel.
	 */
	for (unsigned int i = 1; i < 16; ++i) {
		if (src[i].bits != src[0].bits)
			return false;
	}

	
	
	for (int i = 0; i < 8; i++) {
		dst[i] = 0;
	}

	float src_color_float[3] = {(float)(src->channels.b), (float)(src->channels.g),
				(float)(src->channels.r)};
	union Color base = makeColor555(src_color_float);

	WriteDiff(dst, true);
	WriteFlip(dst, false);
	WriteColors555(dst, base, base);

	uchar best_tbl_idx = 0;
	uchar best_mod_idx = 0;
	uint best_mod_err = 0x7fffffff;

	
	
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		
		


		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			const union Color color = makeColor(base, lum);

			uint mod_err = getColorError(*src, color);
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
 * @brief Orchestrates full 4x4 block compression.
 * Algorithm: Selects flip mode and color quantization scheme based on sub-block averages.
 */
unsigned long compressBlock(__global uchar* dst, const union Color* ver_src, const union Color* hor_src,
												   unsigned long threshold)
{
		uchar g_idx_to_num[4][8] = {
			{0, 4, 1, 5, 2, 6, 3, 7},        
			{8, 12, 9, 13, 10, 14, 11, 15},  
			{0, 4, 8, 12, 1, 5, 9, 13},      
			{2, 6, 10, 14, 3, 7, 11, 15}     
		};

	unsigned long solid_error = 0;
	if (tryCompressSolidBlock(dst, ver_src, &solid_error)) {
		return solid_error;
	}

	const union Color* sub_block_src[4] = {ver_src, ver_src + 8, hor_src, hor_src + 8};

	union Color sub_block_avg[4];
	bool use_differential[2] = {true, true};

	
	
	/**
	 * Block Logic: Sub-block average calculation and differential check.
	 * Invariant: Checks if sub-block average components are within 3-bit delta range.
	 */
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
			sub_block_err[i] += getColorError(sub_block_avg[i], sub_block_src[i][j]);
		}
	}

	// Logic: Heuristic selection between vertical and horizontal sub-block orientation.
	bool flip =
	sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1];

	
	
	for (int i  = 0; i < 8; i++)
		dst[i] = 0;

	WriteDiff(dst, use_differential[!!flip]);
	WriteFlip(dst, flip);

	uchar sub_block_off_0 = flip ? 2 : 0;
	uchar sub_block_off_1 = sub_block_off_0 + 1;

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

/**
 * @brief OpenCL kernel entry point for texture compression.
 * Thread Indexing: 2D grid mapping global work IDs to texture coordinates.
 */
__kernel void compress_k(const int width, const int height, __global uchar *src,
				__global uchar *dst)
{
	int y = get_global_id(0);
	int x = get_global_id(1);


	union Color ver_blocks[16]; 
	union Color hor_blocks[16];

	__local union Color row0[4]; 
	__local union Color row1[4];
	__local union Color row2[4];
	__local union Color row3[4];

	int depl_src = y * 16 * width + (x * 16);

	/**
	 * Block Logic: Shared memory staging.
	 * Logic: Prefetches blocks into local memory to minimize global memory contention.
	 */
	for (int i  = 0; i < 4; i++) {
		int depl = depl_src; 
		row0[i].channels.b = src[depl + (i * 4)];
		row0[i].channels.g = src[depl + (i * 4) + 1];
		row0[i].channels.r = src[depl + (i * 4) + 2];
		row0[i].channels.a = src[depl + (i * 4) + 3];
	}

	for (int i  = 0; i < 4; i++) {
		int depl = depl_src + (width * 4) + (i * 4);
		row1[i].channels.b = src[depl];
		row1[i].channels.g = src[depl + 1];
		row1[i].channels.r = src[depl + 2];
		row1[i].channels.a = src[depl + 3];
	}

	for (int i  = 0; i < 4; i++) {
		int depl = depl_src + 2 * (width * 4) + (i * 4);
		row2[i].channels.b = src[depl];
		row2[i].channels.g = src[depl + 1];
		row2[i].channels.r = src[depl + 2];
		row2[i].channels.a = src[depl + 3];
	}

	for (int i  = 0; i < 4; i++) {
		int depl = depl_src + 3 * (width * 4) + (i * 4);
		row3[i].channels.b = src[depl];
		row3[i].channels.g = src[depl + 1];
		row3[i].channels.r = src[depl + 2];
		row3[i].channels.a = src[depl + 3];
	}

	
	/**
	 * Block Logic: Block layout reformatting.
	 * Logic: Rearranges staging data to satisfy internal orientation requirements.
	 */
	for (int i = 0; i < 2; i++) {
		ver_blocks[i].channels.b = row0[i].channels.b;
		ver_blocks[i].channels.g = row0[i].channels.g;
		ver_blocks[i].channels.r = row0[i].channels.r;
		ver_blocks[i].channels.a = row0[i].channels.a;
		ver_blocks[2 + i].channels.b = row1[i].channels.b;
		ver_blocks[2 + i].channels.g = row1[i].channels.g;
		ver_blocks[2 + i].channels.r = row1[i].channels.r;
		ver_blocks[2 + i].channels.a = row1[i].channels.a;
		ver_blocks[4 + i].channels.b = row2[i].channels.b;
		ver_blocks[4 + i].channels.g = row2[i].channels.g;
		ver_blocks[4 + i].channels.r = row2[i].channels.r;
		ver_blocks[4 + i].channels.a = row2[i].channels.a;
		ver_blocks[6 + i].channels.b = row3[i].channels.b;
		ver_blocks[6 + i].channels.g = row3[i].channels.g;
		ver_blocks[6 + i].channels.r = row3[i].channels.r;
		ver_blocks[6 + i].channels.a = row3[i].channels.a;

		ver_blocks[8 + i].channels.b = row0[2 + i].channels.b;
		ver_blocks[8 + i].channels.g = row0[2 + i].channels.g;
		ver_blocks[8 + i].channels.r = row0[2 + i].channels.r;
		ver_blocks[8 + i].channels.a = row0[2 + i].channels.a;
		ver_blocks[10 + i].channels.b = row1[2 + i].channels.b;
		ver_blocks[10 + i].channels.g = row1[2 + i].channels.g;
		ver_blocks[10 + i].channels.r = row1[2 + i].channels.r;
		ver_blocks[10 + i].channels.a = row1[2 + i].channels.a;
		ver_blocks[12 + i].channels.b = row2[2 + i].channels.b;
		ver_blocks[12 + i].channels.g = row2[2 + i].channels.g;
		ver_blocks[12 + i].channels.r = row2[2 + i].channels.r;
		ver_blocks[12 + i].channels.a = row2[2 + i].channels.a;
		ver_blocks[14 + i].channels.b = row3[2 + i].channels.b;
		ver_blocks[14 + i].channels.g = row3[2 + i].channels.g;
		ver_blocks[14 + i].channels.r = row3[2 + i].channels.r;
		ver_blocks[14 + i].channels.a = row3[2 + i].channels.a;
	}

	
	for (int i = 0; i < 4; i++) {
		ver_blocks[i].channels.b = row0[i].channels.b;
		ver_blocks[i].channels.g = row0[i].channels.g;
		ver_blocks[i].channels.r = row0[i].channels.r;
		ver_blocks[i].channels.a = row0[i].channels.a;
		ver_blocks[4 + i].channels.b = row1[i].channels.b;
		ver_blocks[4 + i].channels.g = row1[i].channels.g;
		ver_blocks[4 + i].channels.r = row1[i].channels.r;
		ver_blocks[4 + i].channels.a = row1[i].channels.a;
		ver_blocks[8 + i].channels.b = row2[i].channels.b;
		ver_blocks[8 + i].channels.g = row2[i].channels.g;
		ver_blocks[8 + i].channels.r = row2[i].channels.r;
		ver_blocks[8 + i].channels.a = row2[i].channels.a;
		ver_blocks[12 + i].channels.b = row3[i].channels.b;
		ver_blocks[12 + i].channels.g = row3[i].channels.g;
		ver_blocks[12 + i].channels.r = row3[i].channels.r;
		ver_blocks[12 + i].channels.a = row3[i].channels.a;

	}
	int dst_depl = (y * (width / 4) + x) * 8;
	compressBlock(dst + dst_depl, ver_blocks, hor_blocks, 0xffffffff);


}

// ... rest of the host-side OpenCL management code ...
