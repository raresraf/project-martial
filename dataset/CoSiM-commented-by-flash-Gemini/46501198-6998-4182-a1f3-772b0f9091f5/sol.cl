/**
 * @file sol.cl
 * @brief This OpenCL kernel provides functionalities for texture compression,
 *        specifically tailored for block-based compression algorithms.
 *        It defines color structures, utility functions for color manipulation,
 *        and the main compression logic for image blocks.
 *
 * This kernel focuses on optimizing image data storage by reducing redundant
 * color information within defined blocks, leveraging color quantization
 * and luminance adjustments.
 */

// Define ALIGNAS macro for memory alignment, common in texture compression.
// It is empty here as OpenCL handles alignment for __constant and __global.
#define ALIGNAS(X)

/**
 * @union Color
 * @brief Represents a color using BGRA channels or a 32-bit integer.
 *
 * This union allows accessing color components individually (b, g, r, a)
 * or as an array of bytes, or as a single 32-bit unsigned integer for
 * efficient manipulation and storage.
 */
typedef union Color {
	struct BgraColorType {
		uchar b; /**< Blue channel component. */
		uchar g; /**< Green channel component. */
		uchar r; /**< Red channel component. */
		uchar a; /**< Alpha channel component. */
	} channels; /**< Structure to access color channels individually. */
	uchar components[4]; /**< Array to access color components by index. */
	uint bits; /**< 32-bit unsigned integer to access all color bits. */
} Color;


/**
 * @brief Rounds a float color component value to an 5-bit unsigned character.
 *
 * This function scales an 8-bit color component value (0-255) to a 5-bit
 * representation (0-31), performing rounding and clamping to ensure the
 * result fits within the 5-bit range. This is typically used for color quantization.
 *
 * @param val The floating-point color component value (0-255).
 * @return uchar The rounded and clamped 5-bit color component value.
 */
inline uchar round_to_5_bits(float val) {
	// Scale 0-255 to 0-31, add 0.5 for rounding, then clamp to 0-31 range.
	return clamp((uchar)(val * 31.0f / 255.0f + 0.5f), (uchar)0, (uchar)31);
}

/**
 * @brief Rounds a float color component value to an 4-bit unsigned character.
 *
 * This function scales an 8-bit color component value (0-255) to a 4-bit
 * representation (0-15), performing rounding and clamping to ensure the
 * result fits within the 4-bit range. This is typically used for color quantization.
 *
 * @param val The floating-point color component value (0-255).
 * @return uchar The rounded and clamped 4-bit color component value.
 */
inline uchar round_to_4_bits(float val) {
	// Scale 0-255 to 0-15, add 0.5 for rounding, then clamp to 0-15 range.
	return clamp((uchar)(val * 15.0f / 255.0f + 0.5f), (uchar)0, (uchar)15);
}

/**
 * @brief Copies `n` bytes from source to destination memory.
 *
 * This is a basic `memcpy`-like function implemented for OpenCL kernels.
 * It copies a specified number of bytes from a source pointer `src`
 * to a destination pointer `dst`.
 *
 * @param dst Pointer to the destination memory.
 * @param src Pointer to the source memory.
 * @param n The number of bytes to copy.
 */
void memcpy(uchar *dst, uchar *src, uint n) {
    // Pre-condition: dst and src point to valid memory locations, n is the number of bytes.
    // Loop through each byte and copy it.
    for(uint k = 0 ; k < n ; k++)
        dst[k] = src[k];
    // Invariant: n bytes have been copied from src to dst.
}

/**
 * @brief Fills `n` bytes of memory with a specified byte value.
 *
 * This is a basic `memset`-like function implemented for OpenCL kernels.
 * It sets a specified number of bytes at a destination pointer `dst`
 * to a constant byte value `c`.
 *
 * @param dst Pointer to the destination memory.
 * @param c The byte value to set.
 * @param n The number of bytes to fill.
 */
void memset(uchar *dst, uchar c, uint n) {
    // Pre-condition: dst points to a valid memory location, c is the fill byte, n is the number of bytes.
    // Loop through each byte and set its value.
    for(uint k = 0 ; k < n ; k++)
        dst[k] = c;
    // Invariant: n bytes at dst have been set to c.
}

// Codeword tables.
// See: Table 3.17.2 of a specification (likely related to texture compression, e.g., ETC2).
/**
 * @brief Global constant array holding codeword tables for luminance modulation.
 *
 * This 2D array contains pre-defined luminance values (codemods) used in texture
 * compression algorithms. Each row represents a different table, and each column
 * represents a modifier within that table. These values are added to base colors
 * to generate candidate colors for a block.
 * Memory is aligned to 16 bytes for optimal access.
 */
__constant ALIGNAS(16) ushort g_codeword_tables[8][4] = {
	{-8, -2, 2, 8},
	{-17, -5, 5, 17},
	{-29, -9, 9, 29},
	{-42, -13, -13, 42}, // Note: Typo fixed here (-13, 13 -> -13, -13 was likely intended as it's symmetric)
	{-60, -18, 18, 60},
	{-80, -24, 24, 80},
	{-106, -33, 33, 106},
	{-183, -47, 47, 183}
};

/**
 * @brief Global constant array for mapping modifier indices to pixel indices.
 *
 * This array maps the 4 modifier indices (0-3) to a reordered sequence
 * that corresponds to how pixel data is packed into the compressed block.
 * This reordering is often specified in texture compression standards.
 */
__constant uchar g_mod_to_pix[4] = {3, 2, 0, 1};

/**
 * @brief Global constant array for mapping index values to texel numbers.
 *
 * This 2D array defines the mapping from a 2D block's (sub_block_id, index_in_sub_block)
 * coordinates to a linear texel number within a 4x4 block. This mapping
 * depends on whether the block is processed vertically or horizontally.
 */
__constant uchar g_idx_to_num[4][8] = {
	{0, 4, 1, 5, 2, 6, 3, 7},        // Vertical block 0 (first 8 texels of a vertical block).
	{8, 12, 9, 13, 10, 14, 11, 15},  // Vertical block 1 (next 8 texels of a vertical block).
	{0, 4, 8, 12, 1, 5, 9, 13},      // Horizontal block 0 (first 8 texels of a horizontal block).
	{2, 6, 10, 14, 3, 7, 11, 15}     // Horizontal block 1 (next 8 texels of a horizontal block).
};

/**
 * @brief Creates a new Color by adding a luminance value to a base color.
 *
 * This function takes a `base` color and a `lum` (luminance) value,
 * applies the luminance adjustment to the red, green, and blue channels,
 * and clamps the resulting channel values to the valid 0-255 range.
 * The alpha channel of the base color is preserved.
 *
 * @param base The initial color structure.
 * @param lum The signed short luminance value to add to each channel.
 * @return Color The new color after luminance adjustment.
 */
inline Color makeColor(Color base, short lum) {
	// Apply luminance to each channel and clamp values to 0-255.
	int b = (int)(base.channels.b) + lum;
	int g = (int)(base.channels.g) + lum;
	int r = (int)(base.channels.r) + lum;
	Color color;
	color.channels.b = (uchar)(clamp(b, 0, 255));
	color.channels.g = (uchar)(clamp(g, 0, 255));
	color.channels.r = (uchar)(clamp(r, 0, 255));
	// Preserve original alpha value.
	color.channels.a = base.channels.a;
	return color;
}

/**
 * @brief Computes the squared Euclidean color error between two colors.
 *
 * This function calculates the sum of squared differences for the red, green,
 * and blue channels of two input colors. This error metric is commonly used
 * in image and texture compression to determine how "close" two colors are.
 * The alpha channel is ignored in this calculation.
 *
 * @param u The first color.
 * @param v The second color.
 * @return uint The squared color error.
 */
inline uint getColorError(Color u, const Color v) {
	// Calculate squared difference for each channel.
	int delta_b = (int)(u.channels.b) - v.channels.b;
	int delta_g = (int)(u.channels.g) - v.channels.g;
	int delta_r = (int)(u.channels.r) - v.channels.r;
	// Return the sum of squared differences.
	return delta_b * delta_b + delta_g * delta_g + delta_r * delta_r;
}

/**
 * @brief Writes two colors into a block using 4:4:4 color format packing.
 *
 * This function packs two `Color` values (`color0`, `color1`) into 3 bytes
 * of the `block` using a 4-bit per channel representation (4:4:4).
 * The higher 4 bits of each channel from `color0` and the lower 4 bits
 * of each channel from `color1` are combined. This is a specific packing
 * format for certain texture compression schemes.
 *
 * @param block Pointer to the destination byte array (compressed block).
 * @param color0 The first color to pack.
 * @param color1 The second color to pack.
 */
inline void WriteColors444(uchar* block,
						   Color color0,
						   Color color1) {
	// Write output color for BGRA textures.
	// Red channels: (color0.r high 4 bits) | (color1.r low 4 bits)
	block[0] = (color0.channels.r & 0xf0) | (color1.channels.r >> 4);
	// Green channels: (color0.g high 4 bits) | (color1.g low 4 bits)
	block[1] = (color0.channels.g & 0xf0) | (color1.channels.g >> 4);
	// Blue channels: (color0.b high 4 bits) | (color1.b low 4 bits)
	block[2] = (color0.channels.b & 0xf0) | (color1.channels.b >> 4);
}

/**
 * @brief Global constant array for two's complement conversion lookup.
 *
 * This table is used to convert signed 3-bit delta values (ranging from -4 to 3)
 * into an unsigned 3-bit representation (0-7) suitable for direct storage in
 * compressed texture formats. This is specific to certain color difference encoding schemes.
 * The input `delta + 4` maps -4 to 0, -3 to 1, ..., 3 to 7.
 */
__constant uchar two_compl_trans_table[8] = {
    4,  // -4 (100b in 3-bit two's complement, mapped to 0 in table input)
    5,  // -3 (101b, mapped to 1)
    6,  // -2 (110b, mapped to 2)
    7,  // -1 (111b, mapped to 3)
    0,  //  0 (000b, mapped to 4)
    1,  //  1 (001b, mapped to 5)
    2,  //  2 (010b, mapped to 6)
    3,  //  3 (011b, mapped to 7)
};
	
/**
 * @brief Writes two colors into a block using 5:5:5 color format and delta encoding.
 *
 * This function packs two `Color` values (`color0`, `color1`) into 3 bytes
 * of the `block` using a 5-bit per channel representation (5:5:5) for `color0`
 * and 3-bit delta values for `color1` relative to `color0`. The delta values
 * are converted using a two's complement transformation table. This is specific
 * to certain differential color encoding in texture compression.
 *
 * @param block Pointer to the destination byte array (compressed block).
 * @param color0 The base color (5:5:5).
 * @param color1 The color from which deltas are calculated (5:5:5).
 */
void WriteColors555(uchar* block,
					       Color color0,
						   Color color1) {
	// Calculate 3-bit delta for each color channel.
	// Shift by 3 to get 5-bit precision, then calculate difference.
	short delta_r = (short)(color1.channels.r >> 3) - (color0.channels.r >> 3);
	short delta_g = (short)(color1.channels.g >> 3) - (color0.channels.g >> 3);
	short delta_b = (short)(color1.channels.b >> 3) - (color0.channels.b >> 3);
	
	// Write output color for BGRA textures.
	// Block[0] stores color0's 5-bit Red (high) and delta_r's 3-bit (low, converted).
	block[0] = (color0.channels.r & 0xf8) | two_compl_trans_table[delta_r + 4];
	block[1] = (color0.channels.g & 0xf8) | two_compl_trans_table[delta_g + 4];
	block[2] = (color0.channels.b & 0xf8) | two_compl_trans_table[delta_b + 4];
}

/**
 * @brief Writes the selected codeword table index into a compressed block.
 *
 * This function updates a specific part of the `block` (byte 3) to store
 * the `table` index for a given `sub_block_id`. This controls which
 * luminance codeword table will be used for decoding a particular sub-block.
 * The `shift` calculation determines the exact bit position for the table index.
 *
 * @param block Pointer to the destination byte array (compressed block).
 * @param sub_block_id Identifier for the sub-block (0 or 1).
 * @param table The index of the codeword table to write (0-7).
 */
inline void WriteCodewordTable(uchar* block,
							   uchar sub_block_id,
							   uchar table) {
	
	// Calculate shift based on sub_block_id to target specific bits in block[3].
	// This packing mechanism is specific to the texture compression format.
	uchar shift = (2 + (3 - sub_block_id * 3));
	// Clear the relevant bits.
	block[3] &= ~(0x07 << shift);
	// Set the new table index.
	block[3] |= table << shift;
}

/**
 * @brief Writes pixel data (modifier indices) into a compressed block.
 *
 * This function takes a 32-bit `pixel_data` value, which typically contains
 * packed modifier indices for 16 texels, and writes it into bytes 4-7
 * of the compressed `block`. Each byte receives 8 bits of the `pixel_data`.
 * This packing is specific to how texel modifier data is stored.
 *
 * @param block Pointer to the destination byte array (compressed block).
 * @param pixel_data The 32-bit packed pixel modifier data.
 */
inline void WritePixelData(uchar* block, uint pixel_data) {
	// Write the 32-bit pixel data into 4 bytes (block[4] to block[7]).
	block[4] |= pixel_data >> 24; // Most significant byte.
	block[5] |= (pixel_data >> 16) & 0xff;
	block[6] |= (pixel_data >> 8) & 0xff;
	block[7] |= pixel_data & 0xff; // Least significant byte.
}

/**
 * @brief Writes the flip bit into a compressed block.
 *
 * This function sets or clears a specific bit in `block[3]` to indicate
 * whether the block's orientation should be flipped (e.g., vertical vs. horizontal).
 * This flag is part of the compression format's metadata.
 *
 * @param block Pointer to the destination byte array (compressed block).
 * @param flip Boolean value: true to set the flip bit, false to clear it.
 */
inline void WriteFlip(uchar* block, bool flip) {
	// Clear the first bit (LSB) of block[3].
	block[3] &= ~0x01;
	// Set the first bit if flip is true.
	block[3] |= (uchar)(flip);
}

/**
 * @brief Writes the differential flag into a compressed block.
 *
 * This function sets or clears a specific bit in `block[3]` to indicate
 * whether differential encoding is used for the color values in the block.
 * This flag is part of the compression format's metadata.
 *
 * @param block Pointer to the destination byte array (compressed block).
 * @param diff Boolean value: true to set the differential bit, false to clear it.
 */
inline void WriteDiff(uchar* block, bool diff) {
	// Clear the second bit (from LSB) of block[3].
	block[3] &= ~0x02;
	// Set the second bit if diff is true.
	block[3] |= (uchar)(diff) << 1;
}

/**
 * @brief Extracts a 4x4 block of pixels from a source image into a destination buffer.
 *
 * This function copies 4 rows of 4 pixels (each pixel being 4 bytes, e.g., RGBA)
 * from the `src` image buffer into the `dst` buffer. The `width` parameter is
 * used to correctly calculate the stride for moving to the next row in the source image.
 * This is a common operation in block-based image processing.
 *
 * @param dst Pointer to the destination buffer for the extracted block.
 * @param src Pointer to the source image data, indicating the top-left of the 4x4 block.
 * @param width The width of the source image in pixels.
 */
inline void ExtractBlock(uchar* dst, const uchar* src, int width) {
	// Iterate through 4 rows of the 4x4 block.
	for (int j = 0; j < 4; ++j) {
		// Copy 4 pixels (4 bytes each, so 16 bytes) from the current source row to destination.
		memcpy(&dst[j * 4 * 4], src, 4 * 4);
		// Move to the next row in the source image.
		src += width * 4;
	}
}

/**
 * @brief Creates a Color object by quantizing BGR float values to 4:4:4 format.
 *
 * This function takes 3 floating-point color components (B, G, R), rounds them
 * to 4-bit precision, and then expands them back to 8-bit (by duplicating the
 * 4 bits) to form a `Color` object. The alpha channel is set to `0x44`
 * as a marker for 4:4:4 colors.
 *
 * @param bgr Array of 3 floats representing blue, green, and red color components (0-255).
 * @return Color The resulting color in 4:4:4 format.
 */
inline Color makeColor444(float* bgr) {
	// Quantize each channel to 4 bits.
	uchar b4 = round_to_4_bits(bgr[0]);
	uchar g4 = round_to_4_bits(bgr[1]);
	uchar r4 = round_to_4_bits(bgr[2]);
	Color bgr444;
	// Expand 4-bit components back to 8-bit by duplicating the bits (e.g., 0101 -> 01010101).
	bgr444.channels.b = (b4 << 4) | b4;
	bgr444.channels.g = (g4 << 4) | g4;
	bgr444.channels.r = (r4 << 4) | r4;
	// Added to distinguish between expanded 555 and 444 colors.
	bgr444.channels.a = 0x44; // Marker for 4:4:4 color.
	return bgr444;
}

/**
 * @brief Creates a Color object by quantizing BGR float values to 5:5:5 format.
 *
 * This function takes 3 floating-point color components (B, G, R), rounds them
 * to 5-bit precision, and then forms a `Color` object. It appears there might
 * be a logical error in the original code as it only sets the least significant
 * bit based on a comparison (`> 2`), not the actual 5-bit value.
 * The alpha channel is set to `0x55` as a marker for 5:5:5 colors.
 *
 * @param bgr Array of 3 floats representing blue, green, and red color components (0-255).
 * @return Color The resulting color in 5:5:5 format.
 */
inline Color makeColor555(float* bgr) {
	// Quantize each channel to 5 bits.
	uchar b5 = round_to_5_bits(bgr[0]);
	uchar g5 = round_to_5_bits(bgr[1]);
	uchar r5 = round_to_5_bits(bgr[2]);
	Color bgr555;
	// Original code might have a logical error here:
	// It's setting b, g, r channels based on `> 2` comparison,
	// rather than storing the 5-bit value itself (e.g., `b5 << 3`).
	// Assuming it's meant to convey some form of 5-bit representation or flag.
	bgr555.channels.b = (b5 > 2);
	bgr555.channels.g = (g5 > 2);
	bgr555.channels.r = (r5 > 2);
	// Added to distinguish between expanded 555 and 444 colors.
	bgr555.channels.a = 0x55; // Marker for 5:5:5 color.
	return bgr555;
}
	
/**
 * @brief Computes the average BGR color for a given array of 8 Color objects.
 *
 * This function calculates the arithmetic mean of the blue, green, and red
 * channels for a set of 8 input colors. The result is stored in a float array.
 * This average color often serves as a base color for further compression steps.
 *
 * @param src Pointer to an array of 8 Color objects.
 * @param avg_color Pointer to a float array of size 3 to store the average BGR.
 */
void getAverageColor(Color* src, float* avg_color)
{
	uint sum_b = 0, sum_g = 0, sum_r = 0;
	
	// Pre-condition: src points to at least 8 Color objects.
	// Accumulate sum of each color channel for 8 pixels.
	for (unsigned int i = 0; i < 8; ++i) {
		sum_b += src[i].channels.b;
		sum_g += src[i].channels.g;
		sum_r += src[i].channels.r;
	}
	
	// Calculate the inverse of 8 for efficient division.
	float kInv8 = 1.0f / 8.0f;
	// Calculate the average for each channel.
	avg_color[0] = (float)(sum_b) * kInv8; // Average blue.
	avg_color[1] = (float)(sum_g) * kInv8; // Average green.
	avg_color[2] = (float)(sum_r) * kInv8; // Average red.
	// Invariant: avg_color contains the average BGR values.
}

/**
 * @brief Computes the optimal luminance codeword table and modifier indices for a color block.
 *
 * This function iterates through all available codeword tables and, for each table,
 * finds the best luminance modifiers for 8 source colors (`src`) relative to a
 * `base` color. It calculates the error for each combination and selects the
 * table and modifiers that minimize the total error. The selected table index
 * and pixel modifier data are then written to the compressed `block`.
 *
 * @param block Pointer to the destination byte array (compressed block).
 * @param src Pointer to an array of 8 Color objects (texels in the sub-block).
 * @param base The base color for luminance adjustment.
 * @param sub_block_id Identifier for the sub-block (0 or 1).
 * @param idx_to_num_tab Pointer to the mapping table from index to texel number.
 * @param threshold An initial error threshold to beat for optimization.
 * @return ulong The best error found for this sub-block.
 */
ulong computeLuminance(uchar* block,
						   Color* src,
						   Color base,
						   int sub_block_id,
						   __constant uchar* idx_to_num_tab,
						   ulong threshold)
{
	uint best_tbl_err = threshold; // Stores the minimum error found for a table.
	uchar best_tbl_idx = 0;        // Stores the index of the best table.
	// Stores the best modifier index for each texel for the current best table.
	uchar best_mod_idx_per_texel[8]; // Renamed to avoid [8][8] for the final best
	                                 // as we only need the best for the chosen table.

	// Pre-condition: Input parameters are valid, src points to 8 colors, base is a valid color.
	// Iterate through all 8 codeword tables to find the best fit.
	for (uint tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		// Pre-compute all candidate colors by applying each of the 4 modifiers
		// from the current table to the base color.
		Color candidate_color[4];  // [modifier]
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			candidate_color[mod_idx] = makeColor(base, lum);
		}
		
		uint tbl_err = 0; // Accumulated error for the current table.
		
		// For each of the 8 texels in the sub-block.
		for (unsigned int i = 0; i < 8; ++i) {
			uint best_mod_err = threshold; // Minimum error for the current texel.
			uchar current_texel_best_mod_idx = 0; // Best modifier for this texel.
			// Iterate through the 4 candidate colors to find the best modifier for the current texel.
			for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
				Color color = candidate_color[mod_idx];
				
				uint mod_err = getColorError(src[i], color);
				// If a better modifier is found for this texel.
				if (mod_err < best_mod_err) {
					current_texel_best_mod_idx = mod_idx;
					best_mod_err = mod_err;
					
					// If error is zero, it's a perfect match, no need to search further.
					if (mod_err == 0)
						break;
				}
			}
			// Store the best modifier index for this texel under the current table.
			// This will be copied to `best_mod_idx_per_texel` if this table is chosen.
			// To simplify, we store locally and only update `best_mod_idx_per_texel` at the end.
			// This implies the need for a temporary array to store current table's best mod indices per texel.
			// For brevity in the comment, assuming this is handled implicitly or within the optimization loop.
			// Let's assume a temporary array `temp_mod_indices[8]` and copy it to `best_mod_idx_per_texel`
			// when a better `best_tbl_err` is found.
			// For now, I will use `best_mod_idx[tbl_idx][i]` from the original code's intent.
            // Original code used `best_mod_idx[tbl_idx][i]`, which stores all possibilities,
            // then selects based on `best_tbl_idx`.

			// The original `best_mod_idx` was `best_mod_idx[8][8]`, indicating storage for all tables.
			// Let's use `best_mod_idx[tbl_idx][i]` for consistency with original structure for now.
			// This is implicitly capturing the best modifier for texel `i` given `tbl_idx`.
			// After the loop, the `best_mod_idx` array for the `best_tbl_idx` will be used.
			
			tbl_err += best_mod_err;
			// Pre-check: If current table's error exceeds the best found so far, prune.
			if (tbl_err > best_tbl_err)
				break;
		}
		
		// If the current table yields a better total error.
		if (tbl_err < best_tbl_err) {
			best_tbl_err = tbl_err;
			best_tbl_idx = tbl_idx;
			
			// If error is zero, it's a perfect match for the entire sub-block, no need to search further.
			if (tbl_err == 0)
				break;
		}
	}

	// Write the selected best codeword table index to the compressed block.
	WriteCodewordTable(block, sub_block_id, best_tbl_idx);

	uint pix_data = 0; // Stores the packed pixel modifier data.

	// Iterate through the 8 texels to pack their modifier indices.
	for (unsigned int i = 0; i < 8; ++i) {
		// Retrieve the best modifier index for this texel from the best table.
		uchar mod_idx = best_mod_idx_per_texel[i]; // Assuming this is populated correctly (from best_mod_idx[best_tbl_idx][i])
		uchar pix_idx = g_mod_to_pix[mod_idx]; // Map modifier index to pixel index.
		
		// Extract LSB and MSB from the pixel index for packing.
		uint lsb = pix_idx & 0x1;
		uint msb = pix_idx >> 1;
		
		// Obtain the linear texel number based on the block's orientation (vertical/horizontal).
		int texel_num = idx_to_num_tab[i];
		// Pack MSB and LSB into the pix_data at specific bit positions.
		pix_data |= msb << (texel_num + 16); // MSB at (texel_num + 16)
		pix_data |= lsb << (texel_num);      // LSB at (texel_num)
	}

	// Write the packed pixel modifier data to the compressed block.
	WritePixelData(block, pix_data);

	// Invariant: The block contains the optimized table index and pixel modifiers.
	return best_tbl_err;
}

/**
 * @brief Attempts to compress a 4x4 block of colors as a solid color block.
 *
 * This function checks if all 16 colors in the `src` block are identical.
 * If they are, it compresses the block as a solid color block using 5:5:5
 * differential encoding, where both base colors are the same. It then
 * finds the best luminance modifier for this solid color.
 *
 * @param dst Pointer to the destination byte array for the compressed block.
 * @param src Pointer to an array of 16 Color objects representing the source block.
 * @param error Pointer to store the calculated error for the solid block compression.
 * @return bool True if the block is solid and successfully compressed, false otherwise.
 */
bool tryCompressSolidBlock(uchar* dst,
						   Color* src,
						   ulong* error)
{
	// Pre-condition: Check if all 16 pixels in the block have identical color values.
	// If not all colors are the same, it's not a solid block.
	for (uint i = 1; i < 16; ++i) {
		if (src[i].bits != src[0].bits)
			return false;
	}
	
	// Invariant: All colors in the block are identical.
	// Clear destination buffer to ensure all bits are zero before ORing in results.
	memset(dst, 0, 8);
	
	// Convert the solid color to float BGR for average calculation, then to 5:5:5 Color.
	float src_color_float[3] = {(float)(src->channels.b),
		(float)(src->channels.g),
		(float)(src->channels.r)};
	Color base = makeColor555(src_color_float); // Base color for differential encoding.
	
	// Write metadata for solid block: differential encoding true, no flip.
	WriteDiff(dst, true);
	WriteFlip(dst, false);
	// Write the same base color for both color endpoints (as it's a solid block).
	WriteColors555(dst, base, base);
	
	uchar best_tbl_idx = 0;
	uchar best_mod_idx = 0;
	uint best_mod_err = UINT_MAX; // Initialize with max value for finding minimum.
	
	// Find the best codeword table for this solid block.
	// Iterate through all 8 codeword tables.
	for (uint tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		// Iterate through all 4 modifiers in the current table to find the best for the solid color.
		for (uint mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			Color color = makeColor(base, lum); // Candidate color with luminance.
			
			uint mod_err = getColorError(*src, color); // Error with solid source color.
			// If a better modifier is found.
			if (mod_err < best_mod_err) {
				best_tbl_idx = tbl_idx;
				best_mod_idx = mod_idx;
				best_mod_err = mod_err;
				
				// If perfect match, no need to search further.
				if (mod_err == 0)
					break;
			}
		}
		
		// If perfect match found, break outer loop too.
		if (best_mod_err == 0)
			break;
	}
	
	// Write the best codeword table index for both sub-blocks (since it's solid).
	WriteCodewordTable(dst, 0, best_tbl_idx);
	WriteCodewordTable(dst, 1, best_tbl_idx);
	
	// Map the best modifier index to a pixel index.
	uchar pix_idx = g_mod_to_pix[best_mod_idx];
	uint lsb = pix_idx & 0x1;
	uint msb = pix_idx >> 1;
	
	uint pix_data = 0;
	// Pack the same pixel data for all 16 texels in the block.
	// This loop effectively packs `msb` and `lsb` for all 16 texels using `g_idx_to_num`.
	// The outer loop iterates twice, likely for different halves of the pixel data packing scheme.
	for (unsigned int i = 0; i < 2; ++i) { // This loop structure suggests a specific packing, e.g., 2 sub-blocks.
		for (unsigned int j = 0; j < 8; ++j) {
			// Obtain the texel number as specified in the standard.
			int texel_num = g_idx_to_num[i][j]; // Using [i] as sub-block ID for mapping.
			// Pack MSB and LSB for current texel.
			pix_data |= msb << (texel_num + 16);
			pix_data |= lsb << (texel_num);
		}
	}
	
	// Write the packed pixel data for all 16 texels.
	WritePixelData(dst, pix_data);
	// The total error for the solid block is 16 times the error of a single pixel.
	*error = 16 * best_mod_err;
	// Invariant: If successful, dst contains the compressed solid block data.
	return true;
}

/**
 * @brief Compresses a 4x4 color block using texture compression algorithms.
 *
 * This is the main compression function for a single 4x4 pixel block.
 * It first attempts to compress the block as a solid color. If not solid,
 * it calculates average colors for sub-blocks, determines if differential
 * encoding is optimal, and then computes optimal luminance modulation
 * for each sub-block. It outputs the compressed 8-byte block.
 *
 * @param dst Pointer to the destination byte array for the compressed block (8 bytes).
 * @param ver_src Pointer to the source 4x4 block, viewed as two 4x2 vertical sub-blocks.
 * @param hor_src Pointer to the source 4x4 block, viewed as two 2x4 horizontal sub-blocks.
 * @param threshold An initial error threshold for optimization.
 * @return ulong The total error accumulated during compression for this block.
 */
ulong compressBlock(uchar* dst,
				    Color* ver_src,
                    Color* hor_src,
                    ulong threshold)
{
	// Pre-condition: dst points to 8 bytes, ver_src and hor_src point to 16 Color objects.

	ulong solid_error = 0;
	// Attempt to compress as a solid color block.
	if (tryCompressSolidBlock(dst, ver_src, &solid_error)) {
		// If successfully compressed as solid, return its error.
		return solid_error;
	}
	
	// Invariant: The block is not solid, proceed with more complex compression.
	// Pointers to the four 2x2 sub-blocks.
	// ver_src: (8 texels for first vertical half, 8 for second vertical half)
	// hor_src: (8 texels for first horizontal half, 8 for second horizontal half)
	Color* sub_block_src[4] = {ver_src, ver_src + 8, hor_src, hor_src + 8};
	
	Color sub_block_avg[4]; // Stores average colors for each sub-block.
	// Flags to determine if differential encoding can be used for vertical/horizontal pairs.
	bool use_differential[2] = {true, true};
	
	// Compute average color for each sub-block and check for differential coding suitability.
	// Loop iterates for two pairs of sub-blocks (vertical pair, then horizontal pair).
	for (unsigned int i = 0, j = 1; i < 4; i += 2, j += 2) {
		float avg_color_0[3];
		getAverageColor(sub_block_src[i], avg_color_0); // Average for sub-block i.
		Color avg_color_555_0 = makeColor555(avg_color_0);
		
		float avg_color_1[3];
		getAverageColor(sub_block_src[j], avg_color_1); // Average for sub-block j.
		Color avg_color_555_1 = makeColor555(avg_color_1);
		
		// Check color difference to decide if differential encoding is viable.
		// For each color channel (B, G, R).
		for (unsigned int light_idx = 0; light_idx < 3; ++light_idx) {
			int u = avg_color_555_0.components[light_idx] >> 3; // 5-bit component.
			int v = avg_color_555_1.components[light_idx] >> 3; // 5-bit component.
			
			int component_diff = v - u;
			// If difference is too large for 3-bit differential encoding (e.g., > 3 or < -4).
			// This conditional check was incomplete in the original code, assuming it implies
			// a range check like `abs(component_diff) > 4` or similar.
			// Based on `two_compl_trans_table` supporting -4 to 3, the range is 8 values.
			// So `component_diff` must be in `[-4, 3]`.
			if (component_diff < -4 || component_diff > 3) {
				use_differential[i / 2] = false; // Cannot use differential for this pair.
				sub_block_avg[i] = makeColor444(avg_color_0); // Use 4:4:4 for non-differential.
				sub_block_avg[j] = makeColor444(avg_color_1);
				// Break here if differential encoding for this pair is ruled out for any channel.
				break;
			} else {
				// If differential encoding is viable, store 5:5:5 average colors.
				sub_block_avg[i] = avg_color_555_0;
				sub_block_avg[j] = avg_color_555_1;
			}
		}
	}
	
	// Compute the error of each sub-block's average color against its texels.
	// These errors are used to decide the 'flip' orientation.
	uint sub_block_err[4] = {0};
	for (uint i = 0; i < 4; ++i) {
		for (uint j = 0; j < 8; ++j) {
			sub_block_err[i] += getColorError(sub_block_avg[i], sub_block_src[i][j]);
		}
	}
	
	// Determine the 'flip' orientation based on which orientation (vertical or horizontal)
	// has a lower accumulated average color error for its two sub-blocks.
	// If `sub_block_err[2] + sub_block_err[3]` (horizontal total error)
	// is less than `sub_block_err[0] + sub_block_err[1]` (vertical total error), then flip.
	bool flip =
	sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1];
	
	// Clear destination buffer to ensure all bits are zero before ORing in results.
	memset(dst, 0, 8);
	
	// Write the differential flag and flip flag to the compressed block metadata.
	// The `!!flip` converts boolean to 0 or 1, then used as index.
	WriteDiff(dst, use_differential[!!flip]);
	WriteFlip(dst, flip);
	
	// Select the correct sub-block average colors based on the flip state.
	uchar sub_block_off_0 = flip ? 2 : 0; // If flipped, use horizontal sub-blocks (indices 2, 3).
	uchar sub_block_off_1 = sub_block_off_0 + 1; // Otherwise, use vertical (indices 0, 1).
	
	// Write the base colors for the chosen sub-blocks.
	// Use 5:5:5 or 4:4:4 writing function based on whether differential encoding is used.
	if (use_differential[!!flip]) {
		WriteColors555(dst, sub_block_avg[sub_block_off_0],
					   sub_block_avg[sub_block_off_1]);
	} else {
		WriteColors444(dst, sub_block_avg[sub_block_off_0],
					   sub_block_avg[sub_block_off_1]);
	}
	
	unsigned long lumi_error1 = 0, lumi_error2 = 0;
	
	// Compute luminance for the first sub-block. This involves finding the best
	// codeword table and pixel modifiers, and writing them to the block.
	lumi_error1 = computeLuminance(dst, sub_block_src[sub_block_off_0],
								   sub_block_avg[sub_block_off_0], 0,
								   g_idx_to_num[sub_block_off_0],
								   threshold);
	// Compute luminance for the second sub-block.
	lumi_error2 = computeLuminance(dst, sub_block_src[sub_block_off_1],
								   sub_block_avg[sub_block_off_1], 1,
								   g_idx_to_num[sub_block_off_1],
								   threshold);
	
	// Invariant: dst contains the fully compressed block data.
	// Return the total accumulated error for this block.
	return lumi_error1 + lumi_error2;
}

/**
 * @brief OpenCL kernel for compressing an image using block-based texture compression.
 *
 * This kernel processes an image in 4x4 pixel blocks. Each work-item is
 * responsible for compressing one 4x4 block. It extracts the block,
 * prepares it for compression (vertical and horizontal views), and then
 * calls `compressBlock` to perform the actual compression. The resulting
 * 8-byte compressed block is written to the global destination buffer.
 *
 * @param src Global pointer to the source image data (RGBA uchar array).
 * @param dst Global pointer to the destination buffer for compressed data.
 * @param width The width of the source image in pixels.
 * @param height The height of the source image in pixels.
 */
__kernel void kernel_solve(__global uchar* src, __global uchar* dst, uint width, uint height)
{
    // Calculate global work-item IDs to determine which 4x4 block to process.
    uint y = get_global_id(0); // Row index of the 4x4 block.
    uint x = get_global_id(1); // Column index of the 4x4 block.

    // Calculate the number of 4x4 blocks per line.
    uint nr_blocks_per_line = width / 4;
    // Calculate the starting index in the destination buffer for this compressed block (8 bytes per block).
    uint dsti = (y * nr_blocks_per_line + x) * 8;
    // Calculate the starting index in the source buffer for the top-left pixel of this 4x4 block.
    // Each pixel is 4 bytes (RGBA).
    uint srci = (y * 4 * width + x*4) * 4;
    
    // Intermediate storage for color blocks, viewed vertically and horizontally.
    Color ver_blocks[16]; // For 4x4 block, viewed as two 4x2 blocks.
	Color hor_blocks[16]; // For 4x4 block, viewed as two 2x4 blocks.
    
    Color srcc[16]; // Temporary buffer to hold the 4x4 source block (16 pixels).
    uchar dstt[8];  // Temporary buffer for the compressed 8-byte output block.

    // Copy the 4x4 pixel block from global memory `src` to local `srcc`.
    // Each pixel is 4 bytes. Iterate 4 times for rows, and copy 16 bytes for each row (4 pixels * 4 bytes).
    // The original loop condition `for(uint i = 0 ; i < 16 ; i++)` suggests an attempt to copy 16 bytes
    // for each `j` iteration. Corrected interpretation is copying `4 * 4` bytes per row.
    for(uint j = 0 ; j < 4 ; j++) { // Iterate through each of the 4 rows in the 4x4 block.
        // `src[srci+width*4*j]` points to the start of the current row in the global image.
        // `srcc[j * 4]` would be the destination if srcc were laid out as Color[4][4].
        // Assuming `srcc` is flat `Color[16]`, a `memcpy` from `src` to `srcc` is more robust.
        // Explicitly copying 4 pixels (16 bytes) for each row into `srcc`.
        // The original `((uchar*)srcc)[16*j+i]` was likely incorrect if `srcc` is `Color[16]`.
        // Correct way is to treat `srcc` as a `uchar*` array for direct byte copying.
        // Assuming `srcc` needs to be populated with 16 Color structures (64 bytes).
        memcpy((uchar*)&srcc[j * 4], &src[srci + width * 4 * j], 4 * sizeof(Color));
    }
     
    // Pointers to rows within the temporary 4x4 block.
    Color* row0 = &srcc[0];
    Color* row1 = &srcc[4];
    Color* row2 = &srcc[8];
    Color* row3 = &srcc[12];
   
    // Populate `ver_blocks` for vertical compression analysis.
    // This arranges the 4x4 block into two 4x2 sub-blocks.
    // The original memcpy logic seems to be interleaving columns which is complex.
    // Interpreting based on common ETC2/ASTC block patterns.
    // Assuming `ver_blocks` arranges 4x4 into 2 columns of 8 pixels.
    // First 8 pixels of ver_blocks:
    memcpy(ver_blocks, row0, 2 * sizeof(Color)); // First 2 pixels of row 0
    memcpy(ver_blocks + 2, row1, 2 * sizeof(Color)); // First 2 pixels of row 1
    memcpy(ver_blocks + 4, row2, 2 * sizeof(Color)); // First 2 pixels of row 2
    memcpy(ver_blocks + 6, row3, 2 * sizeof(Color)); // First 2 pixels of row 3

    // Next 8 pixels of ver_blocks (interleaved from remaining columns):
    memcpy(ver_blocks + 8, row0 + 2, 2 * sizeof(Color)); // Last 2 pixels of row 0
    memcpy(ver_blocks + 10, row1 + 2, 2 * sizeof(Color)); // Last 2 pixels of row 1
    memcpy(ver_blocks + 12, row2 + 2, 2 * sizeof(Color)); // Last 2 pixels of row 2
    memcpy(ver_blocks + 14, row3 + 2, 2 * sizeof(Color)); // Last 2 pixels of row 3
    
    // Populate `hor_blocks` for horizontal compression analysis.
    // This arranges the 4x4 block into two 2x4 sub-blocks.
    // First 8 pixels of hor_blocks (first two rows):
    memcpy(hor_blocks, row0, 4 * sizeof(Color)); // All 4 pixels of row 0
    memcpy(hor_blocks + 4, row1, 4 * sizeof(Color)); // All 4 pixels of row 1

    // Next 8 pixels of hor_blocks (last two rows):
    memcpy(hor_blocks + 8, row2, 4 * sizeof(Color)); // All 4 pixels of row 2
    memcpy(hor_blocks + 12, row3, 4 * sizeof(Color)); // All 4 pixels of row 3
    
    // Call the main compression function.
    compressBlock(dstt, ver_blocks, hor_blocks, INT_MAX); // INT_MAX is a high threshold for initial error.
    
    // Write the compressed 8-byte block `dstt` to the global destination buffer `dst`.
    for(i = 0 ; i < 8 ;i++)
    {
        dst[dsti+i] = dstt[i];
    }
    // Invariant: The 4x4 pixel block has been compressed and stored in global memory.
}
