
/**
 * @file kernel_madalin.cl
 * @brief OpenCL kernel for ETC1 texture compression.
 *
 * This kernel implements the Ericsson Texture Compression (ETC1) algorithm, which is a lossy
 * texture compression format for 8-bit RGB data. It compresses 4x4 blocks of pixels into 64 bits.
 *
 * The kernel operates on a 4x4 block of pixels at a time. It divides the block into two 2x4 or 4x2
 * sub-blocks and tries to find the best representation for each sub-block using a base color and
 * a set of luminance modifiers. The algorithm chooses between differential and non-differential
 * coding for the base colors of the two sub-blocks, and also decides whether to flip the sub-blocks
 * (i.e., use a vertical or horizontal split) to achieve the best compression quality.
 *
 * @see https://www.khronos.org/registry/OpenGL/extensions/OES/OES_compressed_ETC1_RGB8_texture.txt
 */

/**
 * @brief A union to represent a 32-bit BGRA color.
 *
 * This union allows accessing the color as a single 32-bit integer, an array of 4 bytes, or
 * as individual BGRA channels.
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
 * @brief Clamps a float value between a minimum and maximum.
 */
float clamputz(float val, float min, float max) {
	return val < min ? min : (val > max ? max : val);
}

/**
 * @brief Rounds a float value in the range [0, 255] to a 5-bit representation.
 */
inline uchar round_to_5_bits(float val) {
	return (uchar)clamputz(val * 31.0f / 255.0f + 0.5f, 0, 31);
}

/**
 * @brief Rounds a float value in the range [0, 255] to a 4-bit representation.
 */
inline uchar round_to_4_bits(float val) {
	return (uchar)clamputz(val * 15.0f / 255.0f + 0.5f, 0, 15);
}

// Codeword tables for luminance modification.
// See Table 3.17.2 in the ETC1 specification.
__constant short g_codeword_tables[8][4] = {
	{-8, -2, 2, 8},
	{-17, -5, 5, 17},
	{-29, -9, 9, 29},
	{-42, -13, 13, 42},
	{-60, -18, 18, 60},
	{-80, -24, 24, 80},
	{-106, -33, 33, 106},
	{-183, -47, 47, 183}};

// Maps modifier indices to pixel index values.
// See Table 3.17.3 in the ETC1 specification.
__constant uchar g_mod_to_pix[4] = {3, 2, 0, 1};

// The ETC1 specification index texels as follows:
// [a][e][i][m]     [ 0][ 4][ 8][12]
// [b][f][j][n]  [ 1][ 5][ 9][13]
// [c][g][k][o]     [ 2][ 6][10][14]
// [d][h][l][p]     [ 3][ 7][11][15]

// However, when extracting sub blocks from BGRA data the natural array
// indexing order ends up different. This table translates from the natural array
// indices to the indices used by the specification.
__constant uchar g_idx_to_num[4][8] = {
	{0, 4, 1, 5, 2, 6, 3, 7},        // Vertical block 0.
	{8, 12, 9, 13, 10, 14, 11, 15},  // Vertical block 1.
	{0, 4, 8, 12, 1, 5, 9, 13},      // Horizontal block 0.
	{2, 6, 10, 14, 3, 7, 11, 15}     // Horizontal block 1.
};

/**
 * @brief Constructs a color from a given base color and a luminance value.
 * @param base The base color.
 * @param lum The luminance value to add to each channel of the base color.
 * @return The resulting color after applying the luminance.
 */
inline union Color makeColor(__const union Color base, short lum) {
	int b = (int)(base.channels.b) + lum;
	int g = (int)(base.channels.g) + lum;
	int r = (int)(base.channels.r) + lum;
	union Color color;
	color.channels.b = (uchar)(clamp(b, 0, 255));
	color.channels.g = (uchar)(clamp(g, 0, 255));
	color.channels.r = (uchar)(clamp(r, 0, 255));
	return color;
}

/**
 * @brief Calculates the error metric for two colors. A small error signals that the
 *        colors are similar to each other, a large error signals the opposite.
 * @param u The first color.
 * @param v The second color.
 * @return The squared error between the two colors.
 */
inline uint getColorError(__const union Color u, __const union Color v) {
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
 * @brief Writes the two 4-bit base colors to the output block.
 * @param block The output block.
 * @param color0 The first base color.
 * @param color1 The second base color.
 */
inline void WriteColors444(uchar* block,
						   __const union Color color0,
						   __const union Color color1) {
	// Write output color for BGRA textures.
	block[0] = (color0.channels.r & 0xf0) | (color1.channels.r >> 4);
	block[1] = (color0.channels.g & 0xf0) | (color1.channels.g >> 4);
	block[2] = (color0.channels.b & 0xf0) | (color1.channels.b >> 4);
}

/**
 * @brief Writes the two 5-bit base colors to the output block, using differential coding.
 * @param block The output block.
 * @param color0 The first base color.
 * @param color1 The second base color.
 */
inline void WriteColors555(uchar* block,
						   __const union Color color0,
						   __const union Color color1) {
	// Table for conversion to 3-bit two complement format.
	__const uchar two_compl_trans_table[8] = {
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
	(short)(color1.channels.r >> 3) - (color0.channels.r >> 3);
	short delta_g =
	(short)(color1.channels.g >> 3) - (color0.channels.g >> 3);
	short delta_b =
	(short)(color1.channels.b >> 3) - (color0.channels.b >> 3);
	
	// Write output color for BGRA textures.
	block[0] = (color0.channels.r & 0xf8) | two_compl_trans_table[delta_r + 4];
	block[1] = (color0.channels.g & 0xf8) | two_compl_trans_table[delta_g + 4];
	block[2] = (color0.channels.b & 0xf8) | two_compl_trans_table[delta_b + 4];
}

/**
 * @brief Writes the codeword table index for a sub-block to the output block.
 * @param block The output block.
 * @param sub_block_id The ID of the sub-block (0 or 1).
 * @param table The index of the codeword table to use.
 */
inline void WriteCodewordTable(uchar* block,
							   uchar sub_block_id,
							   uchar table) {
	
	uchar shift = (2 + (3 - sub_block_id * 3));
	block[3] &= ~(0x07 << shift);
	block[3] |= table << shift;
}

/**
 * @brief Writes the pixel data (luminance modifiers) to the output block.
 * @param block The output block.
 * @param pixel_data The 32-bit pixel data.
 */
inline void WritePixelData(uchar* block, uint pixel_data) {
	block[4] |= pixel_data >> 24;
	block[5] |= (pixel_data >> 16) & 0xff;
	block[6] |= (pixel_data >> 8) & 0xff;
	block[7] |= pixel_data & 0xff;
}

/**
 * @brief Writes the flip bit to the output block.
 * @param block The output block.
 * @param flip `true` for a vertical split, `false` for a horizontal split.
 */
inline void WriteFlip(uchar* block, bool flip) {
	block[3] &= ~0x01;
	block[3] |= (uchar)(flip);
}

/**
 * @brief Writes the differential bit to the output block.
 * @param block The output block.
 * @param diff `true` if differential coding is used, `false` otherwise.
 */
inline void WriteDiff(uchar* block, bool diff) {
	block[3] &= ~0x02;
	block[3] |= (uchar)(diff) << 1;
}

/**
 * @brief Extracts a 4x4 block of pixels from the source image.
 * @param dst The destination buffer for the extracted block.
 * @param src The source image data.
 * @param width The width of the source image in pixels.
 */
inline void ExtractBlock(uchar* dst, const uchar* src, int width) {
	for (int j = 0; j < 4; ++j) {
		for (int i = 0; i < 4 * 4; i++) {
            dst[j * 4 * 4 + i] = src[i];
		}
		src += width * 4;
	}
}

/**
 * @brief Compresses and rounds a BGR888 color to BGR444, then expands it back to BGR888.
 * @param bgr A pointer to an array of 3 floats representing the B, G, and R channels.
 * @return The expanded BGR888 color.
 */
inline union Color makeColor444(const float* bgr) {
	uchar b4 = round_to_4_bits(bgr[0]);
	uchar g4 = round_to_4_bits(bgr[1]);
	uchar r4 = round_to_4_bits(bgr[2]);
	union Color bgr444;
	bgr444.channels.b = (b4 << 4) | b4;
	bgr444.channels.g = (g4 << 4) | g4;
	bgr444.channels.r = (r4 << 4) | r4;
	// Added to distinguish between expanded 555 and 444 colors.
	bgr444.channels.a = 0x44;
	return bgr444;
}

/**
 * @brief Compresses and rounds a BGR888 color to BGR555, then expands it back to BGR888.
 * @param bgr A pointer to an array of 3 floats representing the B, G, and R channels.
 * @return The expanded BGR888 color.
 */
inline union Color makeColor555(const float* bgr) {
	uchar b5 = round_to_5_bits(bgr[0]);
	uchar g5 = round_to_5_bits(bgr[1]);
	uchar r5 = round_to_5_bits(bgr[2]);
	union Color bgr555;
	bgr555.channels.b = (b5 > 2);
	bgr555.channels.g = (g5 > 2);
	bgr555.channels.r = (r5 > 2);
	// Added to distinguish between expanded 555 and 444 colors.
	bgr555.channels.a = 0x55;
	return bgr555;
}
	
/**
 * @brief Computes the average color of a sub-block of 8 pixels.
 * @param src A pointer to the source sub-block.
 * @param avg_color A pointer to an array of 3 floats where the average color will be stored.
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
 * @brief Computes the best luminance modifiers for a sub-block.
 * @param block The output block.
 * @param src The source sub-block.
 * @param base The base color for the sub-block.
 * @param sub_block_id The ID of the sub-block (0 or 1).
 * @param idx_to_num_tab A pointer to the index-to-number mapping table.
 * @param threshold The error threshold for early exit.
 * @return The total error for the sub-block.
 */
unsigned long computeLuminance(uchar* block,
						   __const union Color* src,
						   __const union Color base,
						   int sub_block_id,
						   __constant unsigned char* idx_to_num_tab,
						   unsigned long threshold)
{
	uint best_tbl_err = threshold;
	uchar best_tbl_idx = 0;
	uchar best_mod_idx[8][8];  // [table][texel]

	// Try all codeword tables to find the one giving the best results for this
	// block.
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		// Pre-compute all the candidate colors; combinations of the base color and
		// all available luminance values.
		union Color candidate_color[4];  // [modifier]
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			candidate_color[mod_idx] = makeColor(base, lum);
		}
		
		uint tbl_err = 0;
		for (unsigned int i = 0; i < 8; ++i) {
			// Try all modifiers in the current table to find which one gives the
			// smallest error.
			uint best_mod_err = threshold;
			for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
				__const union Color color = candidate_color[mod_idx];
				
				uint mod_err = getColorError(src[i], color);
				if (mod_err < best_mod_err) {
					best_mod_idx[tbl_idx][i] = mod_idx;
					best_mod_err = mod_err;
					
					if (mod_err == 0)
						break;  // We cannot do any better than this.
				}
			}
			
			tbl_err += best_mod_err;
			if (tbl_err > best_tbl_err)
				break;  // We're already doing worse than the best table so skip.
		}
		
		if (tbl_err < best_tbl_err) {
			best_tbl_err = tbl_err;
			best_tbl_idx = tbl_idx;
			
			if (tbl_err == 0)
				break;  // We cannot do any better than this.
		}
	}

	WriteCodewordTable(block, sub_block_id, best_tbl_idx);

	uint pix_data = 0;

	for (unsigned int i = 0; i < 8; ++i) {
		uchar mod_idx = best_mod_idx[best_tbl_idx][i];
		uchar pix_idx = g_mod_to_pix[mod_idx];
		
		uint lsb = pix_idx & 0x1;
		uint msb = pix_idx >> 1;
		
		// Obtain the texel number as specified in the standard.
		int texel_num = idx_to_num_tab[i];
		pix_data |= msb << (texel_num + 16);
		pix_data |= lsb << (texel_num);
	}

	WritePixelData(block, pix_data);

	return best_tbl_err;
}

/**
 * @brief Tries to compress the block under the assumption that it's a single color block.
 * @param dst The destination buffer for the compressed block.
 * @param src The source 4x4 block.
 * @param error A pointer to store the compression error.
 * @return `true` if the block is a solid color block and was compressed, `false` otherwise.
 */
bool tryCompressSolidBlock(uchar* dst,
						   __const union Color* src,
						   unsigned long* error)
{
	for (unsigned int i = 1; i < 16; ++i) {
		if (src[i].bits != src[0].bits)
			return false;
	}
	
	// Clear destination buffer so that we can "or" in the results.
	for (int i = 0 ; i < 8; i++) {
	    dst[i] = 0;
	}
	
	float src_color_float[3] = {(float)(src->channels.b),
		(float)(src->channels.g),
		(float)(src->channels.r)};
	union Color base = makeColor555(src_color_float);
	
	WriteDiff(dst, true);
	WriteFlip(dst, false);
	WriteColors555(dst, base, base);
	
	uchar best_tbl_idx = 0;
	uchar best_mod_idx = 0;
	uint best_mod_err = 4294967295;
	
	// Try all codeword tables to find the one giving the best results for this
	// block.
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		// Try all modifiers in the current table to find which one gives the
		// smallest error.
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			__const union Color color = makeColor(base, lum);
			
			uint mod_err = getColorError(*src, color);
			if (mod_err < best_mod_err) {
				best_tbl_idx = tbl_idx;
				best_mod_idx = mod_idx;
				best_mod_err = mod_err;
				
				if (mod_err == 0)
					break;  // We cannot do any better than this.
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
			// Obtain the texel number as specified in the standard.
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
 * @brief Compresses a 4x4 block of pixels using the ETC1 algorithm.
 * @param dst The destination buffer for the compressed block (64 bits).
 * @param ver_src The source 4x4 block organized for vertical split.
 * @param hor_src The source 4x4 block organized for horizontal split.
 * @param threshold The error threshold for early exit.
 * @return The total compression error for the block.
 */
unsigned long compressBlock(__global uchar* dst,
                           __const union Color* ver_src,
                           __const union Color* hor_src,
                           unsigned long threshold)
{
	unsigned long solid_error = 0;
	if (tryCompressSolidBlock(dst, ver_src, &solid_error)) {
		return solid_error;
	}
	
	__const union  Color* sub_block_src[4] = {ver_src, ver_src + 8, hor_src, hor_src + 8};
	
	union Color sub_block_avg[4];
	bool use_differential[2] = {true, true};
	
	// Compute the average color for each sub block and determine if differential
	// coding can be used.
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
	
	// Compute the error of each sub block before adjusting for luminance. These
	// error values are later used for determining if we should flip the sub
	// block or not.
	uint sub_block_err[4] = {0};
	for (unsigned int i = 0; i < 4; ++i) {
		for (unsigned int j = 0; j < 8; ++j) {
			sub_block_err[i] += getColorError(sub_block_avg[i], sub_block_src[i][j]);
		}
	}
	
	bool flip =
	sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1];
	
	// Clear destination buffer so that we can "or" in the results.
	for (int z = 0; z < 8; z++) {
	    dst[z] = 0;
	}
	
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
	
	// Compute luminance for the first sub block.
	lumi_error1 = computeLuminance(dst, sub_block_src[sub_block_off_0],
								   sub_block_avg[sub_block_off_0], 0,
								   g_idx_to_num[sub_block_off_0],
								   threshold);
	// Compute luminance for the second sub block.
	lumi_error2 = computeLuminance(dst, sub_block_src[sub_block_off_1],
								   sub_block_avg[sub_block_off_1], 1,
								   g_idx_to_num[sub_block_off_1],
								   threshold);
	
	return lumi_error1 + lumi_error2;
}

void my_memcpy(union Color* dst, const union Color* src, int num) {
    for (int i = 0 ; i < num; i++) {
        dst[i] = src[i];
    }
}

/**
 * @brief The main kernel function for ETC1 compression.
 *
 * This kernel is executed for each 4x4 block of pixels in the source image. It reads the block,
 * organizes the data for both vertical and horizontal splits, and then calls `compressBlock` to
 * perform the compression. The total compression error is accumulated in a global buffer.
 *
 * @param src A pointer to the source image data.
 * @param dst A pointer to the destination buffer for the compressed data.
 * @param width The width of the source image in pixels.
 * @param height The height of the source image in pixels.
 * @param buf_error A buffer to accumulate the compression error.
 */
__kernel void kernel_compress_block( __global uchar* src,
                                    __global uchar* dst,
                                    __global uint* width,
                                    __global uint* height,
                                    __global float* buf_error )
{
	uint gid = get_global_id(0);
	uint lid = get_local_id(1);

    union Color ver_blocks[16];
    union Color hor_blocks[16];

    const union Color* row0 = src + 4 * gid / width[0];        // 4 * x  gid / (width / 4)
    const union Color* row1 = row0 + width;
    const union Color* row2 = row1 + width;
    const union Color* row3 = row2 + width;

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

    atomic_add(buf_error[0], compressBlock(dst, ver_blocks, hor_blocks, 4294967295));
}
