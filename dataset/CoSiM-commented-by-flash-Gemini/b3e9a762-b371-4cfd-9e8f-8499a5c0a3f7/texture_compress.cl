/**
 * @file texture_compress.cl
 * @brief This OpenCL kernel implements a block-based texture compression algorithm.
 *
 * The kernel is designed to compress 4x4 pixel blocks of an image into a smaller
 * fixed-size representation (e.g., 8 bytes per block). It utilizes concepts
 * like color quantization, differential encoding, and luminance modulation
 * to achieve compression while minimizing perceived visual artifacts.
 *
 * Each global work-item processes one 4x4 block of the input image.
 */

// Define maximum values for unsigned and signed 32-bit integers.
#define UINT32_MAX  (0xffffffff) /**< Maximum value for an unsigned 32-bit integer. */
#define INT32_MAX   (2147483647) /**< Maximum value for a signed 32-bit integer. */

/**
 * @struct BgraColorType
 * @brief Represents a color with Blue, Green, Red, and Alpha channels.
 *
 * This structure explicitly defines the byte order for BGRA color components,
 * which is common in many imaging contexts.
 */
typedef struct BgraColorType {
    uchar b; /**< Blue channel component. */
    uchar g; /**< Green channel component. */
    uchar r; /**< Red channel component. */
    uchar a; /**< Alpha channel component. */
} BgraColorType;

/**
 * @union Color
 * @brief Represents a color, allowing access to its channels or raw bit data.
 *
 * This union provides flexible access to color information, either as
 * individual BGRA channels or as a single 32-bit unsigned integer,
 * which can be useful for direct memory manipulation or comparisons.
 */
typedef union Color {
    BgraColorType channels; /**< Structure to access color channels (b, g, r, a). */
    uchar components[4]; /**< Array to access color components by index. */
	uint bits; /**< 32-bit unsigned integer for direct bit manipulation. */
} Color;

/**
 * @brief Clamps a value between a minimum and maximum bound.
 *
 * This function ensures that `val` does not go below `min` or above `max`.
 * It's a common utility in graphics and image processing to keep color
 * component values within a valid range (e.g., 0-255).
 *
 * @param val The value to clamp.
 * @param min The minimum allowed value.
 * @param max The maximum allowed value.
 * @return uchar The clamped value.
 */
uchar clamp2( uchar val,  uchar min,  uchar max) {
	// If val is less than min, return min. Otherwise, if val is greater than max, return max.
	// Otherwise, return val.
	return val < min ? min : (val > max ? max : val);
}

/**
 * @brief Rounds a float color component value to an 5-bit unsigned character.
 *
 * This function scales an 8-bit color component value (0-255) to a 5-bit
 * representation (0-31), performing rounding and clamping to ensure the
 * result fits within the 5-bit range. This is typically used for color quantization
 * in texture compression.
 *
 * @param val The floating-point color component value (0-255).
 * @return uchar The rounded and clamped 5-bit color component value.
 */
uchar round_to_5_bits( float val) {
	// Scale 0-255 to 0-31, add 0.5 for rounding, then clamp to 0-31 range.
	return clamp2(val * 31.0f / 255.0f + 0.5f, 0, 31);
}

/**
 * @brief Rounds a float color component value to an 4-bit unsigned character.
 *
 * This function scales an 8-bit color component value (0-255) to a 4-bit
 * representation (0-15), performing rounding and clamping to ensure the
 * result fits within the 4-bit range. This is typically used for color quantization
 * in texture compression.
 *
 * @param val The floating-point color component value (0-255).
 * @return uchar The rounded and clamped 4-bit color component value.
 */
uchar round_to_4_bits( float val) {
	// Scale 0-255 to 0-15, add 0.5 for rounding, then clamp to 0-15 range.
	return clamp2(val * 15.0f / 255.0f + 0.5f, 0, 15);
}

// Codeword tables.
// See: Table 3.17.2 of the ETC2/EAC specification (likely).
/**
 * @brief Global constant array holding codeword tables for luminance modulation.
 *
 * This 2D array contains pre-defined luminance values (codemods) used in texture
 * compression algorithms. Each row represents a different table, and each column
 * represents a modifier within that table. These values are added to base colors
 * to generate candidate colors for a block.
 * Memory is aligned to 16 bytes for optimal access on some architectures.
 */
__attribute__((aligned(16))) __constant short g_codeword_tables[8][4] = {
	{-8, -2, 2, 8},
	{-17, -5, 5, 17},
	{-29, -9, 9, 29},
	{-42, -13, 13, 42},
	{-60, -18, 18, 60},
	{-80, -24, 24, 80},
	{-106, -33, 33, 106},
	{-183, -47, 47, 183}};

// Maps modifier indices to pixel index values.
// See: Table 3.17.3
// The original code commented out these global constants.
// For functions that use them, they are declared locally.
// __constant uchar g_mod_to_pix[4] = {3, 2, 0, 1};

// __constant uchar g_idx_to_num[4][8] = {
// 	{0, 4, 1, 5, 2, 6, 3, 7},        // Vertical block 0.
// 	{8, 12, 9, 13, 10, 14, 11, 15},  // Vertical block 1.
// 	{0, 4, 8, 12, 1, 5, 9, 13},      // Horizontal block 0.
// 	{2, 6, 10, 14, 3, 7, 11, 15}     // Horizontal block 1.
// };

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
Color makeColor(const Color base, short lum) {
	// Apply luminance to each channel and clamp values to 0-255.
	int b = (int)(base.channels.b) + lum;
	int g = (int)(base.channels.g) + lum;
	int r = (int)(base.channels.r) + lum;
	Color color;
	color.channels.b = (uchar)(clamp2(b, 0, 255));
	color.channels.g = (uchar)(clamp2(g, 0, 255));
	color.channels.r = (uchar)(clamp2(r, 0, 255));
	// Preserve original alpha value.
	color.channels.a = base.channels.a;
	return color;
}

/**
 * @brief Calculates the error metric for two colors.
 *
 * This function computes the squared Euclidean distance between two colors
 * (u and v) in RGB space. A smaller error indicates closer colors.
 * The calculation can optionally use a perceived error metric (weighted
 * sum of squared differences) if `USE_PERCEIVED_ERROR_METRIC` is defined.
 *
 * @param u The first color.
 * @param v The second color.
 * @return uint The calculated color error.
 */
uint getColorError(const Color u, const Color v) {
#ifdef USE_PERCEIVED_ERROR_METRIC
	// Calculate perceived luminance difference using standard weights.
	float delta_b = (float)(u.channels.b) - v.channels.b;
	float delta_g = (float)(u.channels.g) - v.channels.g;
	float delta_r = (float)(u.channels.r) - v.channels.r;
	return (uint)(0.299f * delta_b * delta_b +
				  0.587f * delta_g * delta_g +
				  0.114f * delta_r * delta_r);
#else
	// Calculate simple squared Euclidean distance in RGB space.
	int delta_b = (int)(u.channels.b) - v.channels.b;
	int delta_g = (int)(u.channels.g) - v.channels.g;
	int delta_r = (int)(u.channels.r) - v.channels.r;
	return delta_b * delta_b + 
           delta_g * delta_g + 
           delta_r * delta_r;
#endif
}

/**
 * @brief Writes two colors into a block using 4:4:4 color format packing.
 *
 * This function packs two `Color` values (`color0`, `color1`) into 3 bytes
 * of the `block` using a 4-bit per channel representation (4:4:4).
 * The higher 4 bits of each channel from `color0` and the lower 4 bits
 * of each channel from `color1` are combined. This is a specific packing
 * format for certain texture compression schemes (e.g., ETC2).
 *
 * @param block Pointer to the destination byte array (compressed block).
 * @param color0 The first color to pack.
 * @param color1 The second color to pack.
 */
void WriteColors444(uchar* block,
					const Color color0,
					const Color color1) {
	// Write output color for BGRA textures.
	// Red channels: (color0.r high 4 bits) | (color1.r low 4 bits)
	block[0] = (color0.channels.r & 0xf0) | (color1.channels.r >> 4);
	// Green channels: (color0.g high 4 bits) | (color1.g low 4 bits)
	block[1] = (color0.channels.g & 0xf0) | (color1.channels.g >> 4);
	// Blue channels: (color0.b high 4 bits) | (color1.b low 4 bits)
	block[2] = (color0.channels.b & 0xf0) | (color1.channels.b >> 4);
}

/**
 * @brief Writes two colors into a block using 5:5:5 color format and delta encoding.
 *
 * This function packs two `Color` values (`color0`, `color1`) into 3 bytes
 * of the `block` using a 5-bit per channel representation (5:5:5) for `color0`
 * and 3-bit delta values for `color1` relative to `color0`. The delta values
 * are converted using a two's complement transformation table. This is specific
 * to certain differential color encoding in texture compression (e.g., ETC2).
 *
 * @param block Pointer to the destination byte array (compressed block).
 * @param color0 The base color (5:5:5).
 * @param color1 The color from which deltas are calculated (5:5:5).
 */
void WriteColors555(uchar* block,
					const Color color0,
					const Color color1) {
	// Table for conversion to 3-bit two complement format.
	// Maps signed 3-bit delta values (-4 to 3) to unsigned indices (0-7).
	const uchar two_compl_trans_table[8] = {
		4,  // -4 (100b) -> index 0
		5,  // -3 (101b) -> index 1
		6,  // -2 (110b) -> index 2
		7,  // -1 (111b) -> index 3
		0,  //  0 (000b) -> index 4
		1,  //  1 (001b) -> index 5
		2,  //  2 (010b) -> index 6
		3,  //  3 (011b) -> index 7
	};
	
	// Calculate 3-bit delta for each color channel.
	// Shift by 3 to get 5-bit precision, then calculate difference.
	short delta_r = (short)(color1.channels.r >> 3) - (color0.channels.r >> 3);
	short delta_g = (short)(color1.channels.g >> 3) - (color0.channels.g >> 3);
	short delta_b = (short)(color1.channels.b >> 3) - (color0.channels.b >> 3);
	
	// Write output color for BGRA textures.
	// block[0] stores color0's 5-bit Red (high) and delta_r's 3-bit (low, converted).
	block[0] = (color0.channels.r & 0xf8) | two_compl_trans_table[delta_r + 4];
	block[1] = (color0.channels.g & 0xf8) | two_compl_trans_table[delta_g + 4];
	block[2] = (color0.channels.b & 0xf8) | two_compl_trans_table[delta_b + 4];
}

/**
 * @brief Writes the selected codeword table index into a compressed block.
 *
 * This function updates a specific part of the `block` (byte 3) to store
 * the `table` index for a given `sub_block_id`. This controls which
 * luminance codeword table will be used for decoding a particular sub-block
 * in formats like ETC2. The `shift` calculation determines the exact bit
 * position for the table index.
 *
 * @param block Pointer to the destination byte array (compressed block).
 * @param sub_block_id Identifier for the sub-block (0 or 1).
 * @param table The index of the codeword table to write (0-7).
 */
void WriteCodewordTable( uchar* block,  uchar sub_block_id,  uchar table) {
	// Calculate shift based on sub_block_id to target specific bits in block[3].
	// This packing mechanism is specific to the ETC2 format.
	uchar shift = (2 + (3 - sub_block_id * 3));
	// Clear the relevant bits (3 bits for table index).
	block[3] &= ~(0x07 << shift); // 0x07 is 00000111 binary.
	// Set the new table index.
	block[3] |= table << shift;
}

/**
 * @brief Writes pixel data (modifier indices) into a compressed block.
 *
 * This function takes a 32-bit `pixel_data` value, which typically contains
 * packed modifier indices for 16 texels, and writes it into bytes 4-7
 * of the compressed `block`. Each byte receives 8 bits of the `pixel_data`.
 * This packing is specific to how texel modifier data is stored in
 * formats like ETC2.
 *
 * @param block Pointer to the destination byte array (compressed block).
 * @param pixel_data The 32-bit packed pixel modifier data.
 */
void WritePixelData( uchar* block,  uint pixel_data) {
	// Write the 32-bit pixel data into 4 bytes (block[4] to block[7]).
	block[4] |= pixel_data >> 24; // Most significant byte.
	block[5] |= (pixel_data >> 16) & 0xff;
	block[6] |= (pixel_data >> 8) & 0xff;
	block[7] |= pixel_data & 0xff; // Least significant byte.
}

/**
 * @brief Writes the flip bit into a compressed block's metadata.
 *
 * This function sets or clears a specific bit in `block[3]` to indicate
 * whether the block's orientation should be flipped (e.g., vertical vs. horizontal).
 * This flag is part of the ETC2 compression format's mode information.
 *
 * @param block Pointer to the destination byte array (compressed block).
 * @param flip Boolean value: true to set the flip bit, false to clear it.
 */
void WriteFlip( uchar* block,  bool flip) {
	// Clear the first bit (LSB) of block[3].
	block[3] &= ~0x01;
	// Set the first bit if flip is true.
	block[3] |= (uchar)(flip);
}

/**
 * @brief Writes the differential flag into a compressed block's metadata.
 *
 * This function sets or clears a specific bit in `block[3]` to indicate
 * whether differential encoding is used for the color values in the block.
 * This flag is part of the ETC2 compression format's mode information.
 *
 * @param block Pointer to the destination byte array (compressed block).
 * @param diff Boolean value: true to set the differential bit, false to clear it.
 */
void WriteDiff( uchar* block,  bool diff) {
	// Clear the second bit (from LSB) of block[3].
	block[3] &= ~0x02;
	// Set the second bit if diff is true.
	block[3] |= (uchar)(diff) << 1;
}

/**
 * @brief Compresses BGR888 into BGR444 and expands it for comparison.
 *
 * This function takes 3 floating-point BGR color components, quantizes each
 * to 4 bits, then expands these 4-bit values back to 8 bits by duplicating
 * the 4 bits (e.g., `0101` becomes `01010101`). This expanded 8-bit representation
 * simulates how a 4:4:4 texture would be decompressed by hardware.
 * The alpha channel is set to `0x44` as a marker.
 *
 * @param bgr Pointer to an array of 3 floats representing blue, green, and red color components (0-255).
 * @return Color The resulting color in expanded 4:4:4 format.
 */
Color makeColor444( const float* bgr) {
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
 * @brief Compresses BGR888 into BGR555 and expands it for comparison.
 *
 * This function takes 3 floating-point BGR color components, quantizes each
 * to 5 bits, then forms a `Color` object. The original code has a peculiar
 * way of setting the channels (`> 2`) which likely represents some specific
 * encoding detail rather than a direct 5-bit value. This expanded 8-bit
 * representation simulates how a 5:5:5 texture would be decompressed by hardware.
 * The alpha channel is set to `0x55` as a marker.
 *
 * @param bgr Pointer to an array of 3 floats representing blue, green, and red color components (0-255).
 * @return Color The resulting color in expanded 5:5:5 format.
 */
Color makeColor555( const float* bgr) {
	// Quantize each channel to 5 bits.
	uchar b5 = round_to_5_bits(bgr[0]);
	uchar g5 = round_to_5_bits(bgr[1]);
	uchar r5 = round_to_5_bits(bgr[2]);
	Color bgr555;
	// Original code might have a logical error here:
	// It's setting b, g, r channels based on `> 2` comparison,
	// rather than storing the 5-bit value itself (e.g., `b5 << 3`).
	// This might be specific to a particular hardware's interpretation or a subtle optimization.
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
 * This average color often serves as a base color for further compression steps,
 * particularly in modes that use two base colors for interpolation.
 *
 * @param src Pointer to an array of 8 Color objects.
 * @param avg_color Pointer to a float array of size 3 to store the average BGR.
 */
void getAverageColor( const Color* src,  float* avg_color)
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
	const float kInv8 = 1.0f / 8.0f;
	// Calculate the average for each channel.
	avg_color[0] = (float)(sum_b) * kInv8; // Average blue.
	avg_color[1] = (float)(sum_g) * kInv8; // Average green.
	avg_color[2] = (float)(sum_r) * kInv8; // Average red.
	// Invariant: avg_color contains the average BGR values.
}

/**
 * @brief Computes the optimal luminance codeword table and modifier indices for a color sub-block.
 *
 * This function is critical for ETC2 compression. It iterates through all available
 * codeword tables and, for each table, finds the best luminance modifiers for
 * 8 source colors (`src`) relative to a `base` color. It calculates the error
 * for each combination and selects the table and modifiers that minimize the
 * total error for the sub-block. The selected table index and packed pixel
 * modifier data are then written to the compressed `block`.
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
                       const Color* src,
					   const Color base,
					   int sub_block_id,
					   const uchar* idx_to_num_tab,
					   ulong threshold)
{
	uint best_tbl_err = threshold; // Stores the minimum error found for a table.
	uchar best_tbl_idx = 0;        // Stores the index of the best table.
	uchar best_mod_idx_per_texel[8];  // Stores the best modifier index for each texel for the chosen table.
    // Local constant array for mapping modifier indices to pixel indices.
    // (Originally a global __constant in previous version, now local to this function if not global)
    uchar g_mod_to_pix[4] = {3, 2, 0, 1};

	// Pre-condition: Input parameters are valid, src points to 8 colors, base is a valid color.
	// Try all 8 codeword tables to find the one giving the best results for this sub-block.
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		// Pre-compute all 4 candidate colors by combining the base color with
		// each luminance value from the current codeword table.
		Color candidate_color[4];  // [modifier]
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			candidate_color[mod_idx] = makeColor(base, lum);
		}
		
		uint tbl_err = 0; // Accumulated error for the current table across all 8 texels.
		
		// For each of the 8 texels in the sub-block.
		for (unsigned int i = 0; i < 8; ++i) {
			// Try all 4 modifiers in the current table to find which one gives the
			// smallest error for the current texel.
			uint best_mod_err_for_texel = threshold;
			uchar current_texel_best_mod_idx = 0; // The modifier index for this specific texel.

			for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
				const Color color = candidate_color[mod_idx]; // Candidate color with current modifier.
				
				uint mod_err = getColorError(src[i], color);
				// If a better modifier (lower error) is found for this texel.
				if (mod_err < best_mod_err_for_texel) {
					current_texel_best_mod_idx = mod_idx;
					best_mod_err_for_texel = mod_err;
					
					// If error is zero, it's a perfect match for this texel, no need to search further.
					if (mod_err == 0)
						break;
				}
			}
			// Store the best modifier index found for texel 'i' under the current table.
			best_mod_idx_per_texel[i] = current_texel_best_mod_idx; // Temporarily store in the local array.
			
			tbl_err += best_mod_err_for_texel;
			// Optimization: If the accumulated error for the current table already
			// exceeds the best error found so far, then this table cannot be better.
			// Break early to save computation.
			if (tbl_err > best_tbl_err)
				break;
		}
		
		// If the current table yields a better total error across all 8 texels.
		if (tbl_err < best_tbl_err) {
			best_tbl_err = tbl_err;
			best_tbl_idx = tbl_idx;
			
			// If total error is zero, it's a perfect match for the entire sub-block,
			// no need to search any more tables.
			if (tbl_err == 0)
				break;
		}
	}

	// Write the selected best codeword table index to the compressed block metadata.
	WriteCodewordTable(block, sub_block_id, best_tbl_idx);

	uint pix_data = 0; // Initialize 32-bit variable to pack pixel modifier data.

	// Iterate through the 8 texels to pack their modifier indices into `pix_data`.
	for (unsigned int i = 0; i < 8; ++i) {
		// Retrieve the best modifier index for this texel from the best table.
		uchar mod_idx = best_mod_idx_per_texel[i]; // Get the stored best modifier for this texel.
		uchar pix_idx = g_mod_to_pix[mod_idx]; // Map modifier index to pixel index using local table.
		
		// Extract the least significant bit (LSB) and most significant bit (MSB)
		// from the pixel index for packing into the 32-bit `pix_data`.
		uint lsb = pix_idx & 0x1;
		uint msb = pix_idx >> 1;
		
		// Local constant array for mapping index values to texel numbers.
        uchar g_idx_to_num[4][8] = {
            {0, 4, 1, 5, 2, 6, 3, 7},        // Vertical block 0.
            {8, 12, 9, 13, 10, 14, 11, 15},  // Vertical block 1.
            {0, 4, 8, 12, 1, 5, 9, 13},      // Horizontal block 0.
            {2, 6, 10, 14, 3, 7, 11, 15}     // Horizontal block 1.
        };

		// Obtain the linear texel number within the 4x4 block as specified by the standard.
		// `sub_block_id` acts as the first index for `g_idx_to_num` (0 for vertical, 1 for horizontal).
		int texel_num = g_idx_to_num[sub_block_id][i];
		// Pack MSB and LSB into the `pix_data` at specific bit positions.
		// MSB is shifted by `texel_num + 16` (to target the higher 16 bits of 32-bit `pix_data`).
		// LSB is shifted by `texel_num` (to target the lower 16 bits).
		pix_data |= msb << (texel_num + 16);
		pix_data |= lsb << (texel_num);
	}

	// Write the fully packed 32-bit pixel modifier data to the compressed block.
	WritePixelData(block, pix_data);

	// Invariant: The compressed block now contains the optimal table index and pixel modifiers.
	return best_tbl_err;
}

/**
 * @brief Attempts to compress a 4x4 block of colors assuming it is a solid (single color) block.
 *
 * This function first verifies if all 16 colors in the `src` block are identical.
 * If they are, it proceeds to compress the block using a simplified ETC2 mode for
 * solid blocks (differential encoding with both base colors being the same).
 * It then finds the best luminance modifier for this single solid color and
 * stores it in the compressed `dst` block. If the block is not solid, the function
 * returns false without modifying `dst`.
 *
 * @param dst Pointer to the destination byte array for the compressed block (8 bytes).
 * @param src Pointer to an array of 16 Color objects representing the source block.
 * @param error Pointer to store the calculated total error for the solid block compression.
 * @return bool True if the block is solid and successfully compressed, false otherwise.
 */
bool tryCompressSolidBlock(uchar* dst,
						   const Color* src,
						   unsigned long* error)
{
	// Pre-condition: Check if all 16 pixels in the block have identical color values.
	// If not all colors are the same, it's not a solid block.
	for (unsigned int i = 1; i < 16; ++i) {
		if (src[i].bits != src[0].bits)
			return false; // Not a solid block, bail out.
	}
	
	// Invariant: All colors in the block are identical.
	// Clear destination buffer (8 bytes) so that bits can be "or"ed in reliably.
    for (unsigned int i = 0; i < 8; i++)
        dst[i] = 0; // Equivalent to memset(dst, 0, 8); but explicit for OpenCL.
	
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
	uint best_mod_err = UINT32_MAX; // Initialize with max value for finding minimum error.
    // Local constant arrays for mappings (as they were not global in this version).
	uchar g_mod_to_pix[4] = {3, 2, 0, 1};
    uchar g_idx_to_num[4][8] = {
        {0, 4, 1, 5, 2, 6, 3, 7},        // Vertical block 0.
        {8, 12, 9, 13, 10, 14, 11, 15},  // Vertical block 1.
        {0, 4, 8, 12, 1, 5, 9, 13},      // Horizontal block 0.
        {2, 6, 10, 14, 3, 7, 11, 15}     // Horizontal block 1.
    };

	// Try all 8 codeword tables to find the one giving the best results for this block.
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		// Try all 4 modifiers in the current table to find which one gives the
		// smallest error for the solid color.
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			const Color color = makeColor(base, lum); // Candidate color with luminance.
			
			uint mod_err = getColorError(*src, color); // Error with solid source color (first pixel).
			// If a better modifier (lower error) is found.
			if (mod_err < best_mod_err) {
				best_tbl_idx = tbl_idx;
				best_mod_idx = mod_idx;
				best_mod_err = mod_err;
				
				// If perfect match (error is zero), no need to search further.
				if (mod_err == 0)
					break;
			}
		}
		
		// If perfect match found, break outer loop too.
		if (best_mod_err == 0)
			break;
	}
	
	// Write the best codeword table index for both sub-blocks (since it's solid, both use same).
	WriteCodewordTable(dst, 0, best_tbl_idx);
	WriteCodewordTable(dst, 1, best_tbl_idx);
	
	// Map the best modifier index to a pixel index for packing.
	uchar pix_idx = g_mod_to_pix[best_mod_idx];
	uint lsb = pix_idx & 0x1; // Least significant bit.
	uint msb = pix_idx >> 1;  // Most significant bit.
	
	uint pix_data = 0; // Initialize 32-bit variable to pack pixel modifier data.
	// Pack the same pixel data for all 16 texels in the block.
	// This loop effectively packs `msb` and `lsb` for all 16 texels.
	// The outer loop iterates twice, for example, to cover two 8-texel sub-blocks.
	for (unsigned int i = 0; i < 2; ++i) { // This loop structure suggests a specific packing, e.g., 2 sub-blocks.
		for (unsigned int j = 0; j < 8; ++j) {
			// Obtain the linear texel number within the 4x4 block as specified by the standard.
			int texel_num = g_idx_to_num[i][j];
			// Pack MSB and LSB for current texel into `pix_data`.
			pix_data |= msb << (texel_num + 16);
			pix_data |= lsb << (texel_num);
		}
	}
	
	// Write the fully packed 32-bit pixel modifier data to the compressed block.
	WritePixelData(dst, pix_data);
	// The total error for the solid block is 16 times the error of a single pixel.
	*error = 16 * best_mod_err;
	// Invariant: If successful, dst contains the compressed solid block data.
	return true;
}

/**
 * @brief Compresses a 4x4 color block using the ETC2 compression algorithm (non-solid mode).
 *
 * This is the main compression function for a single 4x4 pixel block when it's
 * not a solid color. It first attempts `tryCompressSolidBlock`. If that fails,
 * it proceeds with a more general ETC2 compression mode. This involves:
 * 1. Calculating average colors for sub-blocks.
 * 2. Determining if differential encoding is optimal based on color differences.
 * 3. Deciding on a block 'flip' orientation (vertical/horizontal) based on error.
 * 4. Writing base colors (either 5:5:5 differential or 4:4:4 non-differential).
 * 5. Computing optimal luminance modulation for each sub-block using `computeLuminance`.
 * It outputs the 8-byte compressed block.
 *
 * @param dst Pointer to the destination byte array for the compressed block (8 bytes).
 * @param ver_src Pointer to the source 4x4 block, viewed as two 4x2 vertical sub-blocks.
 * @param hor_src Pointer to the source 4x4 block, viewed as two 2x4 horizontal sub-blocks.
 * @param threshold An initial error threshold for optimization.
 * @return unsigned long The total error accumulated during compression for this block.
 */
unsigned long compressBlock(uchar* dst,
							const Color* ver_src,
							const Color* hor_src,
							unsigned long threshold)
{
	// Pre-condition: dst points to 8 bytes, ver_src and hor_src point to 16 Color objects.

	unsigned long solid_error = 0;
	// First, attempt to compress the block as a solid color.
	if (tryCompressSolidBlock(dst, ver_src, &solid_error)) {
		// If successfully compressed as solid, return its error and bypass further logic.
		return solid_error;
	}
	
	// Invariant: The block is not solid, proceed with more complex ETC2 compression modes.
	// Pointers to the four 8-texel (2x4 or 4x2) sub-blocks for analysis.
	const Color* sub_block_src[4] = {ver_src, ver_src + 8, hor_src, hor_src + 8};
	
	Color sub_block_avg[4]; // Stores calculated average colors for each of the four sub-blocks.
	// Flags to determine if differential encoding can be used for the two pairs of sub-blocks.
	// `use_differential[0]` for first pair (e.g., vertical), `use_differential[1]` for second (e.g., horizontal).
	bool use_differential[2] = {true, true};
	
	// Compute the average color for each sub-block and determine if differential
	// coding can be used between the two average colors of a pair of sub-blocks.
	// This loop iterates for two pairs: (sub_block_src[0], sub_block_src[1]) and (sub_block_src[2], sub_block_src[3]).
	for (unsigned int i = 0, j = 1; i < 4; i += 2, j += 2) {
		float avg_color_0[3];
		getAverageColor(sub_block_src[i], avg_color_0); // Average for first sub-block in pair.
		Color avg_color_555_0 = makeColor555(avg_color_0); // Quantize to 5:5:5 for differential check.
		
		float avg_color_1[3];
		getAverageColor(sub_block_src[j], avg_color_1); // Average for second sub-block in pair.
		Color avg_color_555_1 = makeColor555(avg_color_1); // Quantize to 5:5:5 for differential check.
		
		// For each color channel (Blue, Green, Red).
		for (unsigned int light_idx = 0; light_idx < 3; ++light_idx) {
			int u = avg_color_555_0.components[light_idx] >> 3; // Extract 5-bit component.
			int v = avg_color_555_1.components[light_idx] >> 3; // Extract 5-bit component.
			
			int component_diff = v - u; // Calculate difference between 5-bit components.
			// Pre-condition: Check if the `component_diff` is within the range
			// supported by the 3-bit two's complement delta encoding [-4, 3].
			// If not, differential encoding cannot be used for this pair.
			if (component_diff < -4 || component_diff > 3) {
				use_differential[i / 2] = false; // Mark this pair as non-differential.
				sub_block_avg[i] = makeColor444(avg_color_0); // Use 4:4:4 for non-differential mode.
				sub_block_avg[j] = makeColor444(avg_color_1);
				// If differential encoding is ruled out for any channel, it's ruled out for the pair.
				break;
			} else {
				// If differential encoding is viable, store the 5:5:5 average colors.
				sub_block_avg[i] = avg_color_555_0;
				sub_block_avg[j] = avg_color_555_1;
			}
		}
	}
	
	// Compute the error of each sub-block's average color against its original texels.
	// These errors are used to decide the 'flip' orientation of the block.
	uint sub_block_err[4] = {0};
	for (unsigned int i = 0; i < 4; ++i) {
		for (unsigned int j = 0; j < 8; ++j) {
			sub_block_err[i] += getColorError(sub_block_avg[i], sub_block_src[i][j]);
		}
	}
	
	// Determine the 'flip' orientation:
	// If the sum of errors for horizontal sub-blocks (`sub_block_err[2]` + `sub_block_err[3]`)
	// is less than the sum of errors for vertical sub-blocks (`sub_block_err[0]` + `sub_block_err[1]`),
	// then the block is "flipped" (horizontal division is preferred).
	bool flip = sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1];
	
	// Clear destination buffer (8 bytes) so that bits can be "or"ed in reliably.
    for (unsigned int i = 0; i < 8; i++)
        dst[i] = 0; // Equivalent to memset(dst, 0, 8); but explicit for OpenCL.

    // Write the differential flag and flip flag to the compressed block metadata.
	WriteDiff(dst, use_differential[!!flip]); // `!!flip` converts boolean to 0 or 1.
	WriteFlip(dst, flip);
	
	// Select the correct sub-block average colors based on the flip state.
	// `sub_block_off_0` will be 0 if `flip` is false (vertical division), or 2 if `flip` is true (horizontal division).
	uchar sub_block_off_0 = flip ? 2 : 0;
	uchar sub_block_off_1 = sub_block_off_0 + 1;
	
	// Write the base colors for the chosen sub-blocks.
	// Use 5:5:5 differential writing function if `use_differential` is true for this pair,
	// otherwise use 4:4:4 non-differential writing function.
	if (use_differential[!!flip]) {
		WriteColors555(dst, sub_block_avg[sub_block_off_0],
					   sub_block_avg[sub_block_off_1]);
	} else {
		WriteColors444(dst, sub_block_avg[sub_block_off_0],
					   sub_block_avg[sub_block_off_1]);
	}
	
	unsigned long lumi_error1 = 0, lumi_error2 = 0;
	// Local constant array for mapping index values to texel numbers.
    uchar g_idx_to_num[4][8] = {
        {0, 4, 1, 5, 2, 6, 3, 7},        // Vertical block 0.
        {8, 12, 9, 13, 10, 14, 11, 15},  // Vertical block 1.
        {0, 4, 8, 12, 1, 5, 9, 13},      // Horizontal block 0.
        {2, 6, 10, 14, 3, 7, 11, 15}     // Horizontal block 1.
    };

	// Compute luminance for the first sub-block. This involves finding the best
	// codeword table and pixel modifiers, and writing them to the block.
	lumi_error1 = computeLuminance(dst, sub_block_src[sub_block_off_0],
								   sub_block_avg[sub_block_off_0], 0, // sub_block_id 0
								   g_idx_to_num[sub_block_off_0],
								   threshold);
	// Compute luminance for the second sub-block.
	lumi_error2 = computeLuminance(dst, sub_block_src[sub_block_off_1],
								   sub_block_avg[sub_block_off_1], 1, // sub_block_id 1
								   g_idx_to_num[sub_block_off_1],
								   threshold);
	
	// Invariant: dst contains the fully compressed block data.
	// Return the total accumulated error for this block.
	return lumi_error1 + lumi_error2;
}

/**
 * @brief Custom `memcpy` implementation for OpenCL.
 *
 * This function copies `n` bytes from a source memory location `src`
 * to a destination memory location `dest`. It's a byte-by-byte copy.
 * Note: For OpenCL, using `__global` or `__local` qualifiers might be
 * necessary for pointers if they refer to device memory. This implementation
 * assumes `__private` or implicitly `__private` memory.
 *
 * @param dest Pointer to the destination memory.
 * @param src Pointer to the source memory.
 * @param n The number of bytes to copy.
 */
void myMemCpy(void *dest, void *src, size_t n)
{
   // Typecast src and dest addresses to (char *) for byte-level access.
   char *csrc = (char *)src;
   char *cdest = (char *)dest;
 
   // Copy contents of src[] to dest[] byte by byte.
   for (int i=0; i<n; i++)
       cdest[i] = csrc[i];
}

/**
 * @brief OpenCL kernel for compressing an image using block-based texture compression.
 *
 * This kernel processes an image in 4x4 pixel blocks. Each global work-item is
 * responsible for compressing one 4x4 block. It extracts the relevant block data
 * from the global source image, prepares it for compression (views for vertical
 * and horizontal sub-blocks), and then calls `compressBlock` to perform the
 * actual compression. The resulting 8-byte compressed block is written
 * to the global destination buffer.
 *
 * @param src Global pointer to the source image data (RGBA uchar array).
 * @param dst Global pointer to the destination buffer for compressed data.
 * @param width The width of the source image in pixels.
 * @param height The height of the source image in pixels.
 */
__kernel void compress(__global uchar* src, __global uchar* dst, int width, int height)
{
	// Get the 2D global ID of the current work-item, which corresponds to the
	// row and column index of the 4x4 block to be processed.
	uint block_row_index = get_global_id(0); // Y-coordinate of the 4x4 block.
	uint block_col_index = get_global_id(1); // X-coordinate of the 4x4 block.

    // Temporary buffers to hold the 4x4 block's pixel data, arranged for
    // vertical and horizontal sub-block analysis respectively.
    Color ver_blocks[16]; // For 4x4 block, viewed as two 4x2 blocks.
	Color hor_blocks[16]; // For 4x4 block, viewed as two 2x4 blocks.
	
	unsigned long compressed_error = 0; // Accumulates compression error (not directly used by kernel for output, but for internal error tracking).
	
	// Calculate the Y-coordinate (in pixels) of the top row of the current 4x4 block.
	int y_pixel_start = block_row_index * 4;
	// Adjust the `src` pointer to point to the start of the row containing the current block.
	// `y_pixel_start * width * 4` calculates the byte offset for the starting row.
    src += (unsigned long)y_pixel_start * width * 4; // Cast to unsigned long to prevent overflow with large dimensions.

	// Calculate the X-coordinate (in pixels) of the left-most column of the current 4x4 block.
    int x_pixel_start = block_col_index * 4;
	// Adjust the `dst` pointer to point to the memory location where the compressed
	// 8-byte block will be written. `x_pixel_start * 8 / 4` is `x_pixel_start * 2`,
	// as each 4x4 block compresses to 8 bytes.
    dst += (unsigned long)block_col_index * 8; // Each block is 8 bytes. `block_col_index` is in terms of blocks.

	// Define pointers to the four rows within the current 4x4 pixel block in global memory.
	// Using `__private` for `Color*` to indicate that `row0`, `row1`, etc. are temporary
	// pointers within the kernel's private address space.
	Color* row0 = (Color*)(__private)(src + (unsigned long)x_pixel_start * 4); // Start of first row of 4x4 block.
	Color* row1 = (Color*)(__private)(src + (unsigned long)x_pixel_start * 4 + width * 4); // Start of second row.
	Color* row2 = (Color*)(__private)(src + (unsigned long)x_pixel_start * 4 + width * 8);  // Start of third row.
	Color* row3 = (Color*)(__private)(src + (unsigned long)x_pixel_start * 4 + width * 12); // Start of fourth row.
	
	// Populate `ver_blocks` for vertical compression analysis.
	// This arranges the 4x4 block's pixels into two 4x2 sub-blocks logically.
	// The `myMemCpy` is used to copy `sizeof(Color)` bytes at a time.
	// Original comments indicate a specific interleaving based on typical ETC2 block modes.
	// First 8 pixels of ver_blocks (first two columns of each row).
	myMemCpy(ver_blocks, row0, 2 * sizeof(Color)); // First 2 pixels of row 0.
	myMemCpy(ver_blocks + 2, row1, 2 * sizeof(Color)); // First 2 pixels of row 1.
	myMemCpy(ver_blocks + 4, row2, 2 * sizeof(Color)); // First 2 pixels of row 2.
	myMemCpy(ver_blocks + 6, row3, 2 * sizeof(Color)); // First 2 pixels of row 3.

	// Next 8 pixels of ver_blocks (last two columns of each row).
	myMemCpy(ver_blocks + 8, row0 + 2, 2 * sizeof(Color)); // Last 2 pixels of row 0.
	myMemCpy(ver_blocks + 10, row1 + 2, 2 * sizeof(Color)); // Last 2 pixels of row 1.
	myMemCpy(ver_blocks + 12, row2 + 2, 2 * sizeof(Color)); // Last 2 pixels of row 2.
	myMemCpy(ver_blocks + 14, row3 + 2, 2 * sizeof(Color)); // Last 2 pixels of row 3.
	
	// Populate `hor_blocks` for horizontal compression analysis.
	// This arranges the 4x4 block's pixels into two 2x4 sub-blocks logically.
	// First 8 pixels of hor_blocks (all pixels of the first two rows).
	myMemCpy(hor_blocks, row0, 4 * sizeof(Color)); // All 4 pixels of row 0.
	myMemCpy(hor_blocks + 4, row1, 4 * sizeof(Color)); // All 4 pixels of row 1.

	// Next 8 pixels of hor_blocks (all pixels of the last two rows).
	myMemCpy(hor_blocks + 8, row2, 4 * sizeof(Color)); // All 4 pixels of row 2.
	myMemCpy(hor_blocks + 12, row3, 4 * sizeof(Color)); // All 4 pixels of row 3.
	
	// Call the main compression function to process the current 4x4 block.
	// `INT32_MAX` is used as an initial high threshold for error comparisons.
	compressed_error += compressBlock((uchar*)dst, ver_blocks, hor_blocks, INT32_MAX);

    // Invariant: The 4x4 pixel block has been compressed into 8 bytes and written to `dst`.
    // The accumulated error is updated.
}