
/**
 * @file dima.cl
 * @brief OpenCL kernel for texture compression using ETC1-like algorithms.
 *
 * This kernel implements various helper functions and the main compression
 * logic for image blocks, focusing on color manipulation, error calculation,
 * and data packing into a compressed format. It includes functions for
 * clamping, color conversion (e.g., to 4-bit or 5-bit color), memory
 * operations, and core compression routines like `compressBlock` and
 * `tryCompressSolidBlock`.
 */

/**
 * @brief Macro for specifying data alignment.
 *
 * This macro is a GCC/Clang specific extension (`__attribute__((aligned(X)))`)
 * used to enforce a minimum alignment for a variable or structure to X bytes.
 * This can be crucial for performance, especially when dealing with vectorized
 * operations or specific hardware requirements in OpenCL kernels.
 *
 * @param X The alignment boundary in bytes.
 */
#define ALIGNAS(X)	__attribute__((aligned(X)))

/**
 * @union Color
 * @brief Represents a color value, allowing access by channels, components, or as a single integer.
 *
 * This union provides flexible ways to interpret a 32-bit color value.
 * It can be accessed as individual BGRA (Blue, Green, Red, Alpha) channels,
 * as an array of 4 unsigned characters (components), or as a single 32-bit
 * unsigned integer (bits). This is particularly useful in graphics programming
 * for efficient manipulation and interpretation of pixel data.
 */
union Color {
    struct BgraColorType {
        uchar b; ///< @brief Blue channel component.
        uchar g; ///< @brief Green channel component.
        uchar r; ///< @brief Red channel component.
        uchar a; ///< @brief Alpha channel component (transparency).
    } channels; ///< @brief Access color components individually by name.
    uchar components[4]; ///< @brief Access color components as an array of unsigned characters.
    uint bits; ///< @brief Access the entire color as a single 32-bit unsigned integer.
};


/**
 * @brief Copies a block of `Color` union data from global to local/private memory.
 *
 * This function provides a basic memory copy operation specifically for arrays
 * of `union Color`. It is used to transfer `len` number of `Color` elements
 * from a source in global memory to a destination in local or private memory.
 *
 * @param dest union Color *: Pointer to the destination memory.
 * @param src __global union Color *: Pointer to the source memory in global address space.
 * @param len int: The number of `Color` elements to copy.
 */
void my_memcpy(union Color *dest, __global union Color *src, int len)

{
    for (int i = 0; i < len; i++)
        dest[i] = src[i];

}


/**
 * @brief Copies a block of `uchar` data from one memory location to another.
 *
 * This function performs a byte-by-byte copy of `len` unsigned characters
 * from a source memory address to a destination memory address. It is a
 * generic memory copy utility for `uchar` arrays.
 *
 * @param dest uchar *: Pointer to the destination memory.
 * @param src uchar *: Pointer to the source memory.
 * @param len int: The number of `uchar` elements (bytes) to copy.
 */
void my_memcpy2(uchar *dest, uchar *src, int len)

{
    for (int i = 0; i < len; i++)
        dest[i] = src[i];

}


/**
 * @brief Fills a block of global memory with a specified byte value.
 *
 * This function sets `len` bytes of memory at the `dest` address in the
 * global address space to the value `val`. It's a basic memory initialization
 * routine.
 *
 * @param dest __global uchar *: Pointer to the destination memory in global address space.
 * @param val uchar: The byte value to fill the memory with.
 * @param len int: The number of bytes to set.
 */
void my_memset(__global uchar *dest, uchar val, int len) {
    for (int i = 0; i < len; i++) {
        dest[i] = val;
    }
}


/**
 * @brief Clamps an integer value within a specified range.
 *
 * This function ensures that an integer `val` stays within the inclusive
 * range defined by `min` and `max`. If `val` is less than `min`, it returns `min`.
 * If `val` is greater than `max`, it returns `max`. Otherwise, it returns `val`.
 *
 * @param val int: The input integer value to clamp.
 * @param min int: The minimum allowed value.
 * @param max int: The maximum allowed value.
 * @return uchar: The clamped value, cast to `uchar`.
 */
uchar clamp3(int val, int min, int max) {
    return val  max ? max : val);
}


/**
 * @brief Clamps an unsigned char value within a specified range.
 *
 * This function ensures that an unsigned char `val` stays within the inclusive
 * range defined by `min` and `max`. If `val` is less than `min`, it returns `min`.
 * If `val` is greater than `max`, it returns `max`. Otherwise, it returns `val`.
 *
 * @param val uchar: The input unsigned char value to clamp.
 * @param min uchar: The minimum allowed value.
 * @param max uchar: The maximum allowed value.
 * @return uchar: The clamped value.
 */
uchar clamp2(uchar val, uchar min, uchar max) {
    return val  max ? max : val);
}

/**
 * @brief Rounds a float color component value to a 5-bit representation.
 *
 * This function scales a float value (assumed to be in the 0-255 range)
 * to a 5-bit integer range (0-31), applies rounding, and then clamps the
 * result to ensure it stays within the valid 5-bit range.
 *
 * @param val float: The input float value (color component).
 * @return uchar: The rounded and clamped 5-bit representation as an unsigned char.
 */
uchar round_to_5_bits(float val) {
    return (uchar)clamp2(val * 31.0f / 255.0f + 0.5f, 0, 31);
}

/**
 * @brief Rounds a float color component value to a 4-bit representation.
 *
 * This function scales a float value (assumed to be in the 0-255 range)
 * to a 4-bit integer range (0-15), applies rounding, and then clamps the
 * result to ensure it stays within the valid 4-bit range.
 *
 * @param val float: The input float value (color component).
 * @return uchar: The rounded and clamped 4-bit representation as an unsigned char.
 */
uchar round_to_4_bits(float val) {
    return (uchar)clamp2(val * 15.0f / 255.0f + 0.5f, 0, 15);
}



/**
 * @brief Global constant array of codeword tables for ETC1 texture compression.
 *
 * This 2D array stores pre-defined luminance (or color difference) values
 * used in the ETC1 texture compression algorithm. Each sub-array represents
 * a codeword table, and the values within are modifiers applied to base colors
 * to generate a palette of colors for a block. These tables are crucial for
 * achieving different shades and tones during compression.
 * See: ETC1 specification, Table 3.17.2
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
 * @brief Global constant array mapping modifier indices to pixel index values.
 *
 * This array is used in the ETC1 compression algorithm to map the selected
 * modifier index (from `g_codeword_tables`) to a specific 2-bit pixel index
 * value that is then packed into the compressed texture block.
 * See: ETC1 specification, Table 3.17.3
 */
__constant uchar g_mod_to_pix[4] = {3, 2, 0, 1};





















/**
 * @brief Global constant array for translating sub-block indices to texel numbers.
 *
 * This 2D array helps in mapping the natural array indexing order of texels
 * within a 4x4 image block (when split into 2x4 vertical or 4x2 horizontal sub-blocks)
 * to the specific texel numbers defined by the ETC1 specification. This is essential
 * for correctly packing pixel data into the compressed texture format.
 * The ETC1 specification indexes texels as follows:
 * [a][e][i][m]     [ 0][ 4][ 8][12]
 * [b][f][j][n]  [ 1][ 5][ 9][13]
 * [c][g][k][o]     [ 2][ 6][10][14]
 * [d][h][l][p]     [ 3][ 7][11][15]
 *
 * However, when extracting sub blocks from BGRA data the natural array
 * indexing order ends up different:
 * vertical0: [a][e][b][f]  horizontal0: [a][e][i][m]
 *            [c][g][d][h]               [b][f][j][n]
 * vertical1: [i][m][j][n]  horizontal1: [c][g][k][o]
 *            [k][o][l][p]               [d][h][l][p]
 *
 * This table translates from the natural array indices in a sub block
 * to the indices (number) used by specification and hardware.
 */
__constant uchar g_idx_to_num[4][8] = {
{0, 4, 1, 5, 2, 6, 3, 7},        ///< @brief Vertical block 0 mapping.
{8, 12, 9, 13, 10, 14, 11, 15},  ///< @brief Vertical block 1 mapping.
{0, 4, 8, 12, 1, 5, 9, 13},      ///< @brief Horizontal block 0 mapping.
{2, 6, 10, 14, 3, 7, 11, 15}     ///< @brief Horizontal block 1 mapping.
};


/**
 * @brief Constructs a new color by applying a luminance adjustment to a base color.
 *
 * This function takes a base `Color` and a `lum` (luminance) value. It adds
 * the luminance to each of the Red, Green, and Blue channels of the base color,
 * and then clamps the resulting channel values to the valid 0-255 range.
 * The alpha channel of the base color is preserved.
 *
 * @param base union Color *: Pointer to the base color to which luminance will be applied.
 * @param lum short: The luminance value to add to the R, G, B channels.
 * @return union Color: A new `Color` union with the adjusted R, G, B channels.
 */
union Color makeColor(union Color *base, short lum) {
	int b = (int)(base->channels.b) + lum;
	int g = (int)(base->channels.g) + lum;
	int r = (int)(base->channels.r) + lum;
	union Color color;
	color.channels.b = (uchar)(clamp3(b, 0, 255));
	color.channels.g = (uchar)(clamp3(g, 0, 255));
	color.channels.r = (uchar)(clamp3(r, 0, 255));
	return color;
}



/**
 * @brief Calculates an error metric between two colors.
 *
 * This function quantifies the difference between two `Color` unions (`u` and `v`).
 * A small error value indicates that the colors are perceptually similar,
 * while a large value suggests a significant difference. The calculation
 * can optionally use a perceived error metric (ifdef USE_PERCEIVED_ERROR_METRIC)
 * or a simpler sum of squared differences for RGB channels.
 *
 * @param u union Color *: Pointer to the first color.
 * @param v union Color *: Pointer to the second color.
 * @return uint: The calculated error metric as an unsigned integer.
 */
uint getColorError(union Color *u, union Color *v) {
#ifdef USE_PERCEIVED_ERROR_METRIC
	float delta_b = (float)(u->channels.b) - v->channels.b;
	float delta_g = (float)(u->channels.g) - v->channels.g;
	float delta_r = (float)(u->channels.r) - v->channels.r;
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
 * @brief Writes two 4-bit per channel colors into a compressed block.
 *
 * This function takes two `Color` unions, extracts their red, green, and blue
 * components (assuming they are already in a 4-bit representation or will be
 * truncated), and packs them into the first three bytes of a compressed texture
 * block. This is typically used for BGRA textures where color components are
 * rounded to 4 bits (e.g., in ETC1 differential mode).
 *
 * @param block __global uchar*: Pointer to the global memory location of the compressed block.
 * @param color0 const union Color *: Pointer to the first color (e.g., base color).
 * @param color1 const union Color *: Pointer to the second color (e.g., differential color).
 */
void WriteColors444(__global uchar* block, const union Color *color0, const union Color *color1) {

	block[0] = (color0->channels.r & 0xf0) | (color1->channels.r >> 4);
	block[1] = (color0->channels.g & 0xf0) | (color1->channels.g >> 4);
	block[2] = (color0->channels.b & 0xf0) | (color1->channels.b >> 4);
}

/**
 * @brief Writes two 5-bit per channel colors (or their differential) into a compressed block.
 *
 * This function is used in ETC1 compression for encoding base colors or their
 * differential values when a 5-bit per channel representation is used.
 * It also uses a `two_compl_trans_table` for converting delta values into
 * a 3-bit two's complement format as required by the ETC1 specification.
 *
 * @param block __global uchar*: Pointer to the global memory location of the compressed block.
 * @param color0 const union Color *: Pointer to the first color (e.g., base color).
 * @param color1 const union Color *: Pointer to the second color (e.g., differential color).
 */
void WriteColors555(__global uchar* block, const union Color *color0, const union Color *color1) {

	const uchar two_compl_trans_table[8] = {
		4,  ///< @brief Maps -4 (100b)
		5,  ///< @brief Maps -3 (101b)
		6,  ///< @brief Maps -2 (110b)
		7,  ///< @brief Maps -1 (111b)
		0,  ///< @brief Maps 0 (000b)
		1,  ///< @brief Maps 1 (001b)
		2,  ///< @brief Maps 2 (010b)
		3,  ///< @brief Maps 3 (011b)
		};

	short delta_r = (short)(color1->channels.r >> 3) - (color0->channels.r >> 3);
	short delta_g = (short)(color1->channels.g >> 3) - (color0->channels.g >> 3);
	short delta_b = (short)(color1->channels.b >> 3) - (color0->channels.b >> 3);

	
	block[0] = (color0->channels.r & 0xf8) | two_compl_trans_table[delta_r + 4];
	block[1] = (color0->channels.g & 0xf8) | two_compl_trans_table[delta_g + 4];
	block[2] = (color0->channels.b & 0xf8) | two_compl_trans_table[delta_b + 4];
}

/**
 * @brief Writes the selected codeword table index into a compressed block.
 *
 * This function packs the `table` index (indicating which of `g_codeword_tables`
 * was chosen for a sub-block) into the third byte of the compressed block.
 * The `sub_block_id` determines the exact bit position within that byte.
 *
 * @param block __global uchar*: Pointer to the global memory location of the compressed block.
 * @param sub_block_id uchar: Identifier for the sub-block (0 or 1), used to determine bit shift.
 * @param table uchar: The index of the chosen codeword table (0-7).
 */
void WriteCodewordTable(__global uchar* block, uchar sub_block_id, uchar table) {
    uchar shift = (2 + (3 - sub_block_id * 3));
    block[3] &= ~(0x07 << shift);
    block[3] |= table << shift;
}

/**
 * @brief Writes pixel index data into a compressed block.
 *
 * This function packs the 16-bit pixel index data (which determines which
 * color from the derived palette is used for each pixel in a 4x4 block)
 * into bytes 4-7 of the compressed texture block.
 *
 * @param block __global uchar*: Pointer to the global memory location of the compressed block.
 * @param pixel_data int: The 16-bit pixel index data, potentially spread across multiple bytes.
 */
void WritePixelData(__global uchar* block, int pixel_data) {
    block[4] |= pixel_data >> 24;
    block[5] |= (pixel_data >> 16) & 0xff;
    block[6] |= (pixel_data >> 8) & 0xff;
    block[7] |= pixel_data & 0xff;
}

/**
 * @brief Writes the flip bit into a compressed block.
 *
 * This function sets a specific bit in the third byte of the compressed block
 * to indicate whether the 4x4 pixel block has been 'flipped' (e.g., to select
 * between vertical or horizontal sub-blocks for color definition in ETC1).
 *
 * @param block __global uchar*: Pointer to the global memory location of the compressed block.
 * @param flip int: A non-zero value to set the flip bit, zero to clear it.
 */
void WriteFlip(__global uchar* block, int flip) {
    block[3] &= ~0x01;
    block[3] |= (uchar)(flip);
}

/**
 * @brief Writes the differential flag into a compressed block.
 *
 * This function sets a specific bit in the third byte of the compressed block
 * to indicate whether the block is using differential color encoding (e.g.,
 * 5-bit base colors with 3-bit differentials) or absolute color encoding (4-bit colors).
 *
 * @param block __global uchar*: Pointer to the global memory location of the compressed block.
 * @param diff int: A non-zero value to set the differential flag, zero to clear it.
 */
void WriteDiff(__global uchar* block, int diff) {


    block[3] &= ~0x02;
    block[3] |= (uchar)(diff) << 1;
}

/**
 * @brief Extracts a 4x4 pixel block from a source image into a destination buffer.
 *
 * This function copies a 4x4 block of pixel data from a wider source image
 * (`src`) into a contiguous destination buffer (`dst`). It assumes 4 bytes per
 * pixel (e.g., RGBA or BGRA) and iterates through rows and columns to extract
 * the block.
 *
 * @param dst uchar*: Pointer to the destination buffer for the 4x4 block.
 * @param src uchar*: Pointer to the starting pixel of the 4x4 block within the wider source image.
 * @param width int: The width of the full source image in pixels.
 */
void ExtractBlock(uchar* dst, uchar* src, int width) {
	for (int j = 0; j < 4; ++j) {
		my_memcpy2(&dst[j * 4 * 4], src, 4 * 4);


		src += width * 4;
	}
}





/**
 * @brief Converts a BGR (float) color into a 4-bit per channel (BGR444) `Color` union.
 *
 * This function takes a BGR color represented by an array of floats,
 * rounds each component to a 4-bit value, and then expands it back to an 8-bit
 * representation (e.g., 0xAB becomes 0xAABB) to simulate how hardware would
 * decompress it. The alpha channel is set to `0x44` to distinguish it from 555 colors.
 *
 * @param bgr float*: Pointer to an array of floats representing the Blue, Green, and Red color components.
 * @return union Color: A `Color` union representing the BGR444 color.
 */
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





/**
 * @brief Converts a BGR (float) color into a 5-bit per channel (BGR555) `Color` union.
 *
 * This function takes a BGR color represented by an array of floats,
 * rounds each component to a 5-bit value, and then expands it back to an 8-bit
 * representation to simulate how hardware would decompress it. The alpha channel
 * is set to `0x55` to distinguish it from 444 colors.
 *
 * @param bgr float*: Pointer to an array of floats representing the Blue, Green, and Red color components.
 * @return union Color: A `Color` union representing the BGR555 color.
 */
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

/**
 * @brief Calculates the average color from an array of `Color` unions.
 *
 * This function iterates through the first 8 `Color` unions in the `src` array,
 * sums their individual Blue, Green, and Red channel components, and then
 * computes the average for each channel. The result is stored in the `avg_color`
 * float array.
 *
 * @param src union Color*: Pointer to an array of `Color` unions.
 * @param avg_color float*: Pointer to a float array of size 3 to store the average BGR components.
 */
void getAverageColor(union Color* src, float* avg_color)
{
uint sum_b = 0, sum_g = 0, sum_r = 0;

	for (uint i = 0; i < 8; ++i) {
		sum_b += src[i].channels.b;
		sum_g += src[i].channels.g;
		sum_r += src[i].channels.r;
}

	float kInv8 = 1.0 / 8.0;
	avg_color[0] = (float)(sum_b) * kInv8;
	avg_color[1] = (float)(sum_g) * kInv8;
	avg_color[2] = (float)(sum_r) * kInv8;
}

/**
 * @brief Computes the best luminance table and pixel modifiers for a color block.
 *
 * This function iterates through all available codeword tables (`g_codeword_tables`)
 * to find the one that minimizes the error when applied to the `src` colors
 * with respect to a `base` color. It also determines the best modifier for
 * each pixel and writes the chosen codeword table and packed pixel data into
 * the `block`.
 *
 * @param block __global uchar*: Pointer to the global memory location of the compressed block.
 * @param src union Color*: Pointer to an array of source colors (e.g., from a sub-block).
 * @param base union Color*: Pointer to the base color for luminance adjustments.
 * @param sub_block_id int: Identifier for the sub-block being processed.
 * @param idx_to_num_tab __constant uchar*: Pointer to the `g_idx_to_num` table for texel mapping.
 * @param threshold ulong: An initial error threshold for optimization.
 * @return ulong: The best (minimum) total error achieved for the sub-block.
 */
ulong computeLuminance(__global uchar* block, union Color* src, union Color* base,
	int sub_block_id, __constant uchar* idx_to_num_tab, ulong threshold)
{
	uint best_tbl_err = threshold;
	uchar best_tbl_idx = 0;
	uchar best_mod_idx[8][8];  

	
	
	for (uint tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		
		
		union Color candidate_color[4];  
		for (uint mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			candidate_color[mod_idx] = makeColor(base, lum);
		}

		uint tbl_err = 0;

		for (uint i = 0; i < 8; ++i) {
			
			
			uint best_mod_err = threshold;
			for (uint mod_idx = 0; mod_idx < 4; ++mod_idx) {
				union Color color = candidate_color[mod_idx];

				uint mod_err = getColorError(&src[i], &color);
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

	for (uint i = 0; i < 8; ++i) {
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
 * @brief Attempts to compress a 4x4 pixel block as a solid color block.
 *
 * This function checks if all pixels within a `src` 4x4 color block are
 * identical (i.e., a solid color block). If they are, it attempts to compress
 * the block using a simplified ETC1 scheme suitable for solid colors,
 * determining the best codeword table and writing the compressed data to `dst`.
 * If the block is not solid, it returns false without modifying `dst`.
 *
 * @param dst __global uchar*: Pointer to the global memory location of the compressed block.
 * @param src union Color*: Pointer to the array of 16 source `Color` unions representing the 4x4 block.
 * @param error ulong*: Pointer to a variable that will store the compression error if successful.
 * @return int: 1 if the block was successfully compressed as solid, 0 otherwise.
 */
int tryCompressSolidBlock(__global uchar* dst, union Color* src, ulong* error)
{
	for (uint i = 1; i < 16; ++i) {
		if (src[i].bits != src[0].bits)
			return 0;
	}

	
	my_memset(dst, 0, 8);

	float src_color_float[3] = {(float)(src->channels.b),
	                            (float)(src->channels.g),
	                            (float)(src->channels.r)};
	union Color base = makeColor555(src_color_float);

	WriteDiff(dst, 1);
	WriteFlip(dst, 0);
	WriteColors555(dst, &base, &base);

	uchar best_tbl_idx = 0;
	uchar best_mod_idx = 0;
	uint best_mod_err = 4294967295;

	
	
	for (uint tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		
		
		for (uint mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			union Color color = makeColor(&base, lum);

			uint mod_err = getColorError(src, &color);
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
	for (uint i = 0; i < 2; ++i) {
		for (uint j = 0; j < 8; ++j) {
		
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
 * @brief Compresses a 4x4 pixel block using the ETC1 algorithm.
 *
 * This function is the core compression routine for a 4x4 pixel block.
 * It first attempts to compress the block as a solid color using `tryCompressSolidBlock`.
 * If that fails, it analyzes the sub-blocks, determines if differential coding
 * is applicable, calculates average colors, and then computes the optimal
 * luminance tables and pixel modifiers for each sub-block, writing the
 * compressed data to `dst`.
 *
 * @param dst __global uchar*: Pointer to the global memory location of the 8-byte compressed block.
 * @param ver_src union Color*: Pointer to the vertically split source colors (first 8 pixels of each 2x4 sub-block).
 * @param hor_src union Color*: Pointer to the horizontally split source colors (first 8 pixels of each 4x2 sub-block).
 * @param threshold ulong: An error threshold used for early exits in luminance computation.
 * @return ulong: The total compression error for the block.
 */
ulong compressBlock(__global uchar* dst, union Color* ver_src,
	union Color* hor_src, ulong threshold)
{


	ulong solid_error = 0;
	if (tryCompressSolidBlock(dst, ver_src, &solid_error)) {
		return solid_error;
	}

	union Color* sub_block_src[4] = {ver_src, ver_src + 8, hor_src, hor_src + 8};

	union Color sub_block_avg[4];
	int use_differential[2] = {1, 1};

	
	
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
			if (component_diff  3) {
				use_differential[i / 2] = 0;
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
			sub_block_err[i] += getColorError(&sub_block_avg[i], &(sub_block_src[i][j]));
		}
	}

	int flip = 0;
	if (sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1])
		flip = 1;

	
	my_memset(dst, 0, 8);

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

	ulong lumi_error1 = 0, lumi_error2 = 0;

	
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
 * @kernel mat_mul
 * @brief OpenCL kernel for performing texture compression on an input image.
 *
 * This kernel processes an input image (`src`) in 4x4 pixel blocks and
 * compresses each block using the ETC1 algorithm, writing the compressed
 * output to `dst`. Each work-item (thread) is responsible for processing
 * a specific 4x4 block of the image.
 *
 * @param src __global uchar*: Pointer to the global memory of the source image data.
 * @param dst __global uchar*: Pointer to the global memory where the compressed image data will be written.
 * @param height int: The height of the original image in pixels.
 * @param width int: The width of the original image in pixels.
 */
__kernel void mat_mul(__global uchar* src, __global uchar* dst, int height, int width) {
	
	int gid_0 = get_global_id(0);
	int gid_1 = get_global_id(1);

	
	int y = gid_0 * 4;
	int x = gid_1 * 4;
	
	
	dst += 8 * gid_0 * width / 4 + 8 * gid_1;
	src += gid_0 * 4 * 4 * width;


	union Color ver_blocks[16];
	union Color hor_blocks[16];

	ulong compressed_error = 0;
	
	
	__global union Color* row0 = (__global union Color*)(src + x * 4);
	__global union Color* row1 = row0 + width;
	__global union Color* row2 = row1 + width;
	__global union Color* row3 = row2 + width;

	
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

	
	compressBlock(dst, ver_blocks, hor_blocks, 4294967295);

}
#include "compress.hpp"

using namespace std;

void read_kernel(string file_name, string &str_kernel)
{
	ifstream in_file(file_name.c_str());
	in_file.open(file_name.c_str());
	
	stringstream str_stream;
	str_stream << in_file.rdbuf();

	str_kernel = str_stream.str();
}

unsigned long gpu_profile_kernel(cl_device_id device, const uint8_t *src, uint8_t *dst,
				int width, int height)
{
	
	cl_int ret;
	cl_context context;
	cl_command_queue cmdQueue;
	cl_program program;
	cl_kernel kernel;
	string kernel_src;

	
	int bufMatSizeA = width * height * 4;
	int bufMatSizeC = width * height * 4 / 8;

	
	int width2 = width / 4;
	int height2 = height / 4;

	
	context = clCreateContext(0, 1, &device, NULL, NULL, &ret);
	
	
	cmdQueue = clCreateCommandQueue(context, device, 0, &ret);
	
	
	
	cl_mem bufMatA = clCreateBuffer(context, CL_MEM_READ_ONLY,
		bufMatSizeA, NULL, &ret);
	
	
	cl_mem bufMatC = clCreateBuffer(context, CL_MEM_READ_WRITE,
		bufMatSizeC, NULL, &ret);
		
	
	clEnqueueWriteBuffer(cmdQueue, bufMatA, CL_TRUE, 0,
		bufMatSizeA, src, 0, NULL, NULL);

	
	read_kernel("dima.cl", kernel_src);
	const char* kernel_c_str = kernel_src.c_str();

	
	program = clCreateProgramWithSource(context, 1,
		(const char **) &kernel_c_str, NULL, &ret);

	
	ret = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

	
	size_t log_size;
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    
    	char *log = (char *) malloc(log_size);
  
    	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
	

	
	kernel = clCreateKernel(program, "mat_mul", &ret);

	
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufMatA);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufMatC);
	clSetKernelArg(kernel, 2, sizeof(int), &height);
	clSetKernelArg(kernel, 3, sizeof(int), &width);

	
	size_t globalSize[2] = {(size_t) height2, (size_t) width2};
	ret = clEnqueueNDRangeKernel(cmdQueue, kernel, 2, NULL,
		globalSize, 0, 0, NULL, NULL);

	clFinish(cmdQueue);

	
	clEnqueueReadBuffer(cmdQueue, bufMatC, CL_TRUE, 0,
		bufMatSizeC, dst, 0, NULL, NULL);

		
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseMemObject(bufMatA);
	clReleaseMemObject(bufMatC);
	clReleaseCommandQueue(cmdQueue);
	clReleaseContext(context);
	
	return 0;

}





unsigned long TextureCompressor::compress(const uint8_t* src,
					  uint8_t* dst,
					  int width,
					  int height)
{
	return gpu_profile_kernel(device, src, dst, width, height);
}


void gpu_find(cl_device_id &device)
{
	
	cl_char contor = 0;
	cl_platform_id platform;
	cl_uint platform_num = 0;
	cl_platform_id* platform_list = NULL;

	cl_uint device_num = 0;
	cl_device_id* device_list = NULL;

	size_t attr_size = 0;
	cl_char* attr_data = NULL;

	
	clGetPlatformIDs(0, NULL, &platform_num);
	platform_list = new cl_platform_id[platform_num];

	
	clGetPlatformIDs(platform_num, platform_list, NULL);
	cout << "Platforms found!: " << platform_num << endl;

	
	for(uint platf=0; platf<platform_num; platf++)
	{
		
		clGetPlatformInfo(platform_list[platf],
				CL_PLATFORM_VENDOR, 0, NULL, &attr_size);
		attr_data = new cl_char[attr_size];

		
		clGetPlatformInfo(platform_list[platf],
				CL_PLATFORM_VENDOR, attr_size, attr_data, NULL);
		cout << "Platform " << platf << " " << attr_data;
		delete[] attr_data;

		
		clGetPlatformInfo(platform_list[platf],
				CL_PLATFORM_VERSION, 0, NULL, &attr_size);
		attr_data = new cl_char[attr_size];

		
		clGetPlatformInfo(platform_list[platf],
				CL_PLATFORM_VERSION, attr_size, attr_data, NULL);
		cout << attr_data << endl;
		delete[] attr_data;

		
		platform = platform_list[platf];

		
		clGetDeviceIDs(platform, 
			CL_DEVICE_TYPE_ALL, 0, NULL, &device_num);

		device_list = new cl_device_id[device_num];

		
		clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL,
			device_num, device_list, NULL);
		cout << "\tDevices found " << device_num  << endl;

		
		for(uint dev=0; dev<device_num; dev++)
		{
			
			clGetDeviceInfo(device_list[dev], CL_DEVICE_NAME,
				0, NULL, &attr_size);
			attr_data = new cl_char[attr_size];

			
			clGetDeviceInfo(device_list[dev], CL_DEVICE_NAME,
				attr_size, attr_data, NULL);
			cout << "\tDevice " << dev << " " << attr_data << " ";
			delete[] attr_data;

				
			cl_char* aux = new cl_char[attr_size];
			clGetDeviceInfo(device_list[dev], CL_DEVICE_NAME,
				attr_size, aux, NULL);

			
			clGetDeviceInfo(device_list[dev], CL_DEVICE_VERSION,
				0, NULL, &attr_size);
			attr_data = new cl_char[attr_size];
			
			
			clGetDeviceInfo(device_list[dev], CL_DEVICE_VERSION,
				attr_size, attr_data, NULL);
			cout << attr_data; 
			delete[] attr_data;

			
			if(strstr((char*)aux, "Tesla") != NULL && contor == 0){
				contor = 1;
				device = device_list[dev];
				cout << " <--- SELECTED ";
			}
			delete[] aux;

			cout << endl;
		}
	}

	delete[] platform_list;
	delete[] device_list;
}


TextureCompressor::TextureCompressor() {
	gpu_find(device);
}

TextureCompressor::~TextureCompressor() { }	
	
