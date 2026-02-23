//
// @3722f929-bc24-4c52-8a90-73670c8cf9c1/compression_device.cl
// @brief This file contains OpenCL kernel code and host-side C++ code for texture compression.
// It defines structures and functions for color manipulation, error calculation, and
// the main compression kernel that processes image blocks.
//
// Algorithm: Block-based texture compression (likely a variant of BC1/DXT1 or similar).
// The compression involves quantizing colors, computing luminance, and encoding pixel data.
//
// HPC & Parallelism: The `compression_kernel` is designed to run on a GPU using OpenCL,
// leveraging parallel processing for image block compression.
//
// Time Complexity: The kernel's complexity is per block, multiplied by the number of blocks.
// Host-side setup involves typical OpenCL overhead.
//
// Space Complexity: Depends on image dimensions, for input/output buffers and intermediate storage.
//

// Union for flexible access to color components (BGRA, array, or single uint).
union Color {
	struct BgraColorType {
		uchar b; // Blue channel component
		uchar g; // Green channel component
		uchar r; // Red channel component
		uchar a; // Alpha channel component
	} channels; // Structure to access individual color channels
	uchar components[4]; // Array to access color components generically
	uint bits; // Full 32-bit representation of the color
};

/**
 * @brief Clamps an integer value within a specified minimum and maximum range.
 * @param val The input integer value.
 * @param min The minimum allowed value.
 * @param max The maximum allowed value.
 * @return The clamped value as an unsigned integer.
 */
inline uint my_clamp(int val, int min, int max) {
	if (val < min) // Pre-condition: Check if value is below minimum.
		return min; // Invariant: Return minimum if `val` is less than `min`.
	else if (val > max) // Pre-condition: Check if value is above maximum.
		return max; // Invariant: Return maximum if `val` is greater than `max`.
	return val; // Invariant: Return original value if within range.
}

/**
 * @brief Rounds an 8-bit color component (0-255) to a 5-bit representation (0-31).
 * @param val The 8-bit color component value.
 * @return The rounded 5-bit color component as an unsigned char.
 */
inline uchar round_to_5_bits(int val) {
	// Scale the 8-bit value to 5-bit range (0-31) and add 0.5 for rounding.
	return (uchar) my_clamp(val * 31.0f / 255.0f + 0.5f, 0, 31);
}

/**
 * @brief Rounds an 8-bit color component (0-255) to a 4-bit representation (0-15).
 * @param val The 8-bit color component value.
 * @return The rounded 4-bit color component as an unsigned char.
 */
inline uchar round_to_4_bits(int val) {
	// Scale the 8-bit value to 4-bit range (0-15) and add 0.5 for rounding.
	return (uchar) my_clamp(val * 15.0f / 255.0f + 0.5f, 0, 15);
}

// Global constant array mapping modifier indices to pixel indices.
__constant short g_mod_to_pix[4] = {3, 2, 0, 1};

/**
 * @brief Creates a new Color union by applying a luminance adjustment to a base color.
 * Note: This function returns a pointer to a local variable `color`, which is unsafe.
 * This should ideally return a `union Color` by value or take a pointer to modify.
 * @param base The base color to which luminance will be applied.
 * @param lum The luminance adjustment value.
 * @return A pointer to a new Color union with adjusted luminance. (Potential error due to local variable address).
 */
inline union Color* makeColor(union Color base, short lum) {
	// Apply luminance adjustment to each color channel.
	int b = (int)base.channels.b + (int)lum;
	int g = (int)base.channels.g + (int)lum;
	int r = (int)base.channels.r + (int)lum;
	union Color* color; // Uninitialized pointer.
	// Clamp the adjusted channel values to the valid 0-255 range.
	color->channels.b = (uchar)(clamp(b, 0, 255));
	color->channels.g = (uchar)(clamp(g, 0, 255));
	color->channels.r = (uchar)(clamp(r, 0, 255));
	return (union Color*) color; // Returns address of local stack variable.
}

/**
 * @brief Calculates the squared color error between two colors.
 * Can use a perceived error metric (weighted sum of squared differences) or
 * a simple sum of squared differences for RGB channels.
 * @param u The first color.
 * @param v The second color.
 * @return The calculated color error.
 */
inline uint getColorError(union Color u, union Color v) {
// Pre-processor directive to select between perceived and simple squared error metric.
#ifdef USE_PERCEIVED_ERROR_METRIC
	// Invariant: Calculates a weighted sum of squared differences (perceived error).
	float delta_b = (float)(u.channels.b) - v.channels.b;
	float delta_g = (float)(u.channels.g) - v.channels.g;
	float delta_r = (float)(u.channels.r) - v.channels.r;
	return (uint)(0.299f * delta_b * delta_b +
								 0.587f * delta_g * delta_g +
								 0.114f * delta_r * delta_r);
#else
	// Invariant: Calculates a simple sum of squared differences for RGB channels.
	int delta_b = (int)(u.channels.b) - v.channels.b;
	int delta_g = (int)(u.channels.g) - v.channels.g;
	int delta_r = (int)(u.channels.r) - v.channels.r;
	return delta_b * delta_b + delta_g * delta_g + delta_r * delta_r;
#endif
}

/**
 * @brief Writes two 4-bit per channel colors into a destination block for compression.
 * This packs the R, G, B channels of two colors into 3 bytes.
 * (e.g., color0.r upper 4 bits, color1.r upper 4 bits in block[0])
 * @param block Pointer to the global memory destination block.
 * @param color0 The first color (source for upper 4 bits of each channel).
 * @param color1 The second color (source for lower 4 bits of each channel).
 */
inline void WriteColors444(__global uchar* block,
						    union Color color0,
						    union Color color1
								) {
	// Block Logic: Packs 4-bit color components into 3 bytes.
	// Red channels: block[0] = (color0.r & 0xF0) | (color1.r >> 4)
	block[0] = (color0.channels.r & 0xf0) | (color1.channels.r >> 4);
	// Green channels: block[1] = (color0.g & 0xF0) | (color1.g >> 4)
	block[1] = (color0.channels.g & 0xf0) | (color1.channels.g >> 4);
	// Blue channels: block[2] = (color0.b & 0xF0) | (color1.b >> 4)
	block[2] = (color0.channels.b & 0xf0) | (color1.channels.b >> 4);
}

/**
 * @brief Writes two 5-bit per channel colors into a destination block, using differential encoding.
 * This packs the base color (5 bits per channel) and 3-bit differentials for the second color.
 * @param block Pointer to the global memory destination block.
 * @param color0 The base color (5-bit representation).
 * @param color1 The second color (used to calculate differentials from color0).
 */
inline void WriteColors555(__global uchar* block,
						   union Color color0,
						   union Color color1) {
	// Lookup table for two's complement transformations for 3-bit differentials.
	uchar two_compl_trans_table[8] = {
		4, // -4 -> 4 (0b100)
		5, // -3 -> 5 (0b101)
		6, // -2 -> 6 (0b110)
		7, // -1 -> 7 (0b111)
		0, // 0 -> 0 (0b000)
		1, // 1 -> 1 (0b001)
		2, // 2 -> 2 (0b010)
		3, // 3 -> 3 (0b011)
	};

	// Calculate 3-bit differentials for each channel (relative to color0, shifted by 3 bits).
	short delta_r =
	(short)(color1.channels.r >> 3) - (color0.channels.r >> 3);
	short delta_g =
	(short)(color1.channels.g >> 3) - (color0.channels.g >> 3);
	short delta_b =
	(short)(color1.channels.b >> 3) - (color0.channels.b >> 3);

	// Block Logic: Store the base color (upper 5 bits) and the differential (lower 3 bits, mapped).
	// Red channel: block[0] = (color0.r & 0xF8) | (mapped delta_r)
	block[0] = (color0.channels.r & 0xf8) | two_compl_trans_table[delta_r + 4];
	// Green channel: block[1] = (color0.g & 0xF8) | (mapped delta_g)
	block[1] = (color0.channels.g & 0xf8) | two_compl_trans_table[delta_g + 4];
	// Blue channel: block[2] = (color0.b & 0xF8) | (mapped delta_b)
	block[2] = (color0.channels.b & 0xf8) | two_compl_trans_table[delta_b + 4];
}

/**
 * @brief Writes the codeword table index for a sub-block into the destination block.
 * @param block Pointer to the global memory destination block.
 * @param sub_block_id Identifier for the sub-block (0 to 3).
 * @param table The 3-bit codeword table index to write.
 */
inline void WriteCodewordTable(__global uchar* block,
							   uchar sub_block_id,
							   uchar table) {
	// Calculate the bit shift for the table based on sub_block_id.
	uchar shift = (2 + (3 - sub_block_id * 3));
	// Clear the existing bits for the codeword table.
	block[3] &= ~(0x07 << shift);
	// Write the new codeword table index.
	block[3] |= table << shift;
}

/**
 * @brief Writes encoded pixel data into the destination block.
 * The pixel data is a 32-bit unsigned integer containing 2-bit indices for each pixel.
 * @param block Pointer to the global memory destination block.
 * @param pixel_data The 32-bit encoded pixel data.
 */
inline void WritePixelData(__global uchar* block, uint pixel_data) {
	// Block Logic: Distributes 32-bit pixel data into bytes 4-7 of the block.
	block[4] = (pixel_data >> 24) & 0xff; // Most significant byte to block[4]
	block[5] = (pixel_data >> 16) & 0xff;
	block[6] = (pixel_data >> 8) & 0xff;
	block[7] = pixel_data & 0xff; // Least significant byte to block[7]
}

/**
 * @brief Writes the flip bit into the destination block.
 * This bit indicates if the block's orientation has been flipped.
 * @param block Pointer to the global memory destination block.
 * @param flip Boolean value (true for flipped, false otherwise).
 */
inline void WriteFlip(__global uchar* block, bool flip) {
	// Clear the existing flip bit (LSB of block[3]).
	block[3] &= ~0x01;
	// Set the flip bit based on the boolean value.
	block[3] |= (uchar)(flip);
}

/**
 * @brief Writes the differential encoding bit into the destination block.
 * This bit indicates if differential encoding is used for colors.
 * @param block Pointer to the global memory destination block.
 * @param diff Boolean value (true for differential, false otherwise).
 */
inline void WriteDiff(__global uchar* block, bool diff) {
	// Clear the existing differential bit (second LSB of block[3]).
	block[3] &= ~0x02;
	// Set the differential bit based on the boolean value.
	block[3] |= (uchar)(diff) << 1;
}

/**
 * @brief Performs a byte-by-byte memory copy.
 * @param dst Pointer to the destination memory.
 * @param src Pointer to the source memory.
 * @param width The number of bytes to copy.
 */
inline void memcpy(uchar *dst, uchar *src, int width) {
	// Invariant: Copies `width` bytes from `src` to `dst`.
	for (int i = 0; i < width; i++) {
		dst[i] = src[i];
	}
}

/**
 * @brief Creates a Color union from BGR float values, rounded to 4-bit per channel.
 * @param bgr Array of float values for Blue, Green, Red.
 * @return A Color union with 4-bit per channel color data.
 */
inline union Color makeColor444(float* bgr) {
	// Round BGR components to 4 bits.
	uchar b4 = round_to_4_bits(bgr[0]);
	uchar g4 = round_to_4_bits(bgr[1]);
	uchar r4 = round_to_4_bits(bgr[2]);
	union Color bgr444;
	// Duplicate 4-bit value to fill 8 bits for each channel (e.g., 0xAB -> 0xAABB).
	bgr444.channels.b = (b4 << 4) | b4;
	bgr444.channels.g = (g4 << 4) | g4;
	bgr444.channels.r = (r4 << 4) | r4;
	// Placeholder alpha channel value.
	bgr444.channels.a = 0x44;
	return bgr444;
}

/**
 * @brief Creates a Color union from BGR float values, rounded to 5-bit per channel.
 * @param bgr Array of float values for Blue, Green, Red.
 * @return A Color union with 5-bit per channel color data.
 */
inline union Color makeColor555(float* bgr) {
	// Round BGR components to 5 bits.
	uchar b5 = round_to_5_bits(bgr[0]);
	uchar g5 = round_to_5_bits(bgr[1]);
	uchar r5 = round_to_5_bits(bgr[2]);
	union Color bgr555;
	// Assign the 5-bit values directly. The lower 3 bits are implicitly zeroed by uchar assignment.
	// This appears to be an error in the original code, as (b5 > 2) evaluates to a boolean (0 or 1)
	// instead of assigning the 5-bit value. It should likely be `bgr555.channels.b = b5 << 3;` or similar.
	bgr555.channels.b = (b5 > 2); // Suspicious: assigns boolean result (0 or 1)
	bgr555.channels.g = (g5 > 2); // Suspicious: assigns boolean result (0 or 1)
	bgr555.channels.r = (r5 > 2); // Suspicious: assigns boolean result (0 or 1)
	// Placeholder alpha channel value.
	bgr555.channels.a = 0x55;
	return bgr555;
}

/**
 * @brief Computes the average BGR color from an array of Color unions.
 * @param src Pointer to an array of Color unions (typically 8 colors for a sub-block).
 * @param avg_color Array of 3 floats to store the average B, G, R components.
 */
void getAverageColor(union Color* src, float* avg_color)
{
	// Accumulators for sum of B, G, R components.
	uint sum_b = 0, sum_g = 0, sum_r = 0;

	// Loop over 8 colors to sum their components.
	for (uint i = 0; i < 8; ++i) {
		sum_b += src[i].channels.b;
		sum_g += src[i].channels.g;
		sum_r += src[i].channels.r;
	}

	// Calculate average by dividing by 8.
	float kInv8 = 1.0f / 8.0f;
	avg_color[0] = (float)(sum_b) * kInv8;
	avg_color[1] = (float)(sum_g) * kInv8;
	avg_color[2] = (float)(sum_r) * kInv8;
}

/**
 * @brief Fills a block of memory with a specified byte value.
 * @param dst Pointer to the global memory destination.
 * @param value The byte value to fill with.
 * @param size The number of bytes to fill.
 */
void memset(__global uchar* dst, int value, int size) {
	// Invariant: Fills `size` bytes of `dst` with `value`.
	for (int i = 0; i < size; i++) {
		dst[i] = value;
	}
}

/**
 * @brief Computes the best luminance codeword table and pixel indices for a sub-block.
 * This function iterates through predefined codeword tables to find the one that minimizes
 * the total color error for the given source colors.
 * @param block Pointer to the global memory destination block where table and pixel data will be written.
 * @param src Pointer to the source Color unions (8 colors for a sub-block).
 * @param base The base color from which luminance adjustments are made.
 * @param sub_block_id The identifier for the current sub-block.
 * @param idx_to_num_tab Lookup table for pixel numbering within the block.
 * @param threshold An initial error threshold; tables exceeding this will be discarded.
 * @return The minimum total color error found for the best table.
 */
unsigned long computeLuminance(__global uchar* block,
						   union Color* src,
						   union Color base,
						   int sub_block_id,
						   uchar* idx_to_num_tab,
						   unsigned long threshold)
{
	// Predefined luminance codeword tables. Each table has 4 luminance values.
	short g_codeword_tables[8][4] = {
	{-8, -2, 2, 8},
	{-17, -5, 5, 17},
	{-29, -9, 9, 29},
	{-42, -13, 13, 42},
	{-60, -18, 18, 60},
	{-80, -24, 24, 80},
	{-106, -33, 33, 106},
	{-183, -47, 47, 183}};

	uint best_tbl_err = threshold; // Initialize best table error with a high threshold.
	uchar best_tbl_idx = 0; // Index of the best codeword table found.
	// 2D array to store the best modifier index for each source color for each table.
	uchar best_mod_idx[8][8];

	// Block Logic: Iterate through each codeword table to find the best fit.
	for (uint tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		// Calculate the 4 candidate colors for the current codeword table based on the base color and luminance modifiers.
		union Color candidate_color[4];
		for (uint mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			candidate_color[mod_idx] = *makeColor(base, lum); // Note: makeColor returns a potentially invalid pointer.
		}

		uint tbl_err = 0; // Total error for the current table.

		// For each source color in the sub-block, find the closest candidate color from the current table.
		for (uint i = 0; i < 8; ++i) {
			uint best_mod_err = threshold; // Initialize best modifier error for the current source color.

			// Invariant: Find the modifier index that results in the minimum error for src[i].
			for (uint mod_idx = 0; mod_idx < 4; ++mod_idx) {
				union Color color = candidate_color[mod_idx];

				uint mod_err = getColorError(src[i], color);
				// If current modifier error is better, update best_mod_err and store the mod_idx.
				if (mod_err < best_mod_err) {
					best_mod_idx[tbl_idx][i] = mod_idx;
					best_mod_err = mod_err;

					if (mod_err == 0) // Optimization: If error is 0, this is a perfect match, no need to check further.
						break;
				}
			}

			tbl_err += best_mod_err; // Accumulate error for the current table.
			if (tbl_err > best_tbl_err) // Optimization: If current table error exceeds the best found so far, break.
				break;
		}

		// If the current table's total error is better than the best found so far, update.
		if (tbl_err < best_tbl_err) {
			best_tbl_err = tbl_err;
			best_tbl_idx = tbl_idx;

			if (tbl_err == 0) // Optimization: If error is 0, this is a perfect table, no need to check further.
				break;
		}
	}

	// Write the index of the best codeword table into the output block.
	WriteCodewordTable(block, sub_block_id, best_tbl_idx);

	uint pix_data = 0; // Initialize 32-bit pixel data.

	// For each source color, encode its modifier index into 2-bit pixel data.
	for (unsigned int i = 0; i < 8; ++i) {
		uchar mod_idx = best_mod_idx[best_tbl_idx][i]; // Get the best modifier index.
		uchar pix_idx = g_mod_to_pix[mod_idx]; // Map modifier index to pixel index using global constant.

		uint lsb = pix_idx & 0x1; // Extract least significant bit.
		uint msb = pix_idx >> 1;  // Extract most significant bit.

		// Determine the position for the current pixel in the 32-bit pixel data.
		int texel_num = idx_to_num_tab[i];

		// Invariant: Combine MSB and LSB into `pix_data` at specific bit positions.
		pix_data |= msb << (texel_num + 16); // Store MSB at higher bits.
		pix_data |= lsb << (texel_num);     // Store LSB at lower bits.
	}

	// Write the final encoded pixel data into the output block.
	WritePixelData(block, pix_data);

	return best_tbl_err; // Return the minimum error achieved.
}

/**
 * @brief Attempts to compress a block as a solid color block.
 * A solid color block implies all pixels within the block are identical or very similar.
 * @param dst Pointer to the global memory destination block.
 * @param src Pointer to the source Color unions (16 colors for a 4x4 block).
 * @param error Pointer to an unsigned long where the calculated error for the solid block will be stored.
 * @return True if the block is considered solid and compressed, false otherwise.
 */
bool tryCompressSolidBlock(__global uchar* dst,
						   union Color* src,
						   unsigned long* error)
{
	// Predefined luminance codeword tables (duplicate of g_codeword_tables, consider refactoring).
	short g_codeword_tables[8][4] = {
	{-8, -2, 2, 8},
	{-17, -5, 5, 17},
	{-29, -9, 9, 29},
	{-42, -13, 13, 42},
	{-60, -18, 18, 60},
	{-80, -24, 24, 80},
	{-106, -33, 33, 106},
	{-183, -47, 47, 183}};

	// Pixel numbering lookup table for two 8-pixel sub-blocks (likely for 4x4 blocks).
	uchar g_idx_to_num[4][8] = {
	{0, 4, 1, 5, 2, 6, 3, 7},        // Sub-block 0
	{8, 12, 9, 13, 10, 14, 11, 15},  // Sub-block 1
	{0, 4, 8, 12, 1, 5, 9, 13},      // Sub-block 2 (alternative scan order)
	{2, 6, 10, 14, 3, 7, 11, 15}     // Sub-block 3 (alternative scan order)
	};

	// Block Logic: Check if all 16 pixels in the block have the same color.
	for (unsigned int i = 1; i < 16; ++i) {
		if (src[i].bits != src[0].bits) // Pre-condition: If any pixel differs from the first, it's not a solid block.
			return false; // Invariant: If not solid, return false immediately.
	}

	// If the block is solid, initialize the destination block to zeros.
	memset(dst, 0, 8);

	// Convert the solid color (src[0]) to float BGR components.
	float src_color_float[3] = {(float)(src->channels.b),
		(float)(src->channels.g),
		(float)(src->channels.r)};
	// Convert the float BGR to a 5-bit per channel Color union.
	union Color base = makeColor555(src_color_float);

	// Write compression flags for a solid block: differential encoding true, no flip.
	WriteDiff(dst, true);
	WriteFlip(dst, false);
	// Write the base color (twice, as there's only one color for a solid block).
	WriteColors555(dst, base, base);

	uchar best_tbl_idx = 0; // Index of the best luminance table.
	uchar best_mod_idx = 0; // Index of the best luminance modifier.
	uint best_mod_err = UINT_MAX; // Minimum error found for a single modifier.

	// Block Logic: Find the best luminance modifier from all tables for the solid color.
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			union Color* color = makeColor(base, lum); // Note: makeColor returns a potentially invalid pointer.

			uint mod_err = getColorError(*src, *color);
			// If current modifier error is better, update best_mod_err and store indices.
			if (mod_err < best_mod_err) {
				best_tbl_idx = tbl_idx;
				best_mod_idx = mod_idx;
				best_mod_err = mod_err;

				if (mod_err == 0) // Optimization: Perfect match found.
					break;
			}
		}

		if (best_mod_err == 0) // Optimization: If perfect match for modifier, break outer loop.
			break;
	}

	// Write the best codeword table index for both sub-blocks (since it's a solid block).
	WriteCodewordTable(dst, 0, best_tbl_idx);
	WriteCodewordTable(dst, 1, best_tbl_idx);

	uchar pix_idx = g_mod_to_pix[best_mod_idx]; // Map modifier index to pixel index.
	uint lsb = pix_idx & 0x1; // Extract LSB.
	uint msb = pix_idx >> 1;  // Extract MSB.

	uint pix_data = 0; // Initialize 32-bit pixel data.
	// Block Logic: Fill pixel data for the solid block.
	// Invariant: Each of the 16 pixels will have the same 2-bit encoded value.
	for (unsigned int i = 0; i < 2; ++i) { // Loop twice, potentially for two sub-blocks.
		for (unsigned int j = 0; j < 8; ++j) { // Loop for 8 pixels in each sub-block.
			int texel_num = g_idx_to_num[i][j]; // Get pixel number.
			pix_data |= msb << (texel_num + 16); // Set MSB.
			pix_data |= lsb << (texel_num);     // Set LSB.
		}
	}

	// Write the encoded pixel data.
	WritePixelData(dst, pix_data);
	// Calculate and store the total error for the solid block (16 pixels * best_mod_err).
	*error = 16 * best_mod_err;
	return true; // Return true as the block was successfully compressed as solid.
}


/**
 * @brief Main function to compress a 4x4 color block using texture compression techniques.
 * It first tries solid block compression, then proceeds with more complex methods
 * involving sub-blocks, differential encoding, and flipping.
 * @param dst Pointer to the global memory destination block (8 bytes for compressed data).
 * @param ver_src Pointer to the source Color unions for vertical scan (16 colors).
 * @param hor_src Pointer to the source Color unions for horizontal scan (16 colors).
 * @param threshold Error threshold for compression; not directly used in the current implementation.
 * @return The total error after compression, or 0 if successful (or if not fully implemented).
 */
ulong compressBlock(__global uchar* dst,
										union Color* ver_src,
										union Color* hor_src,
										ulong threshold) {

	unsigned long solid_error = 0;
	// Pre-condition: Attempt to compress as a solid block first.
	if (tryCompressSolidBlock(dst, ver_src, &solid_error)) {
		return solid_error; // Invariant: If solid, return its error and finish.
	}

	// Pixel numbering lookup table for two 8-pixel sub-blocks (likely for 4x4 blocks).
	uchar g_idx_to_num[4][8] = {
		{0, 4, 1, 5, 2, 6, 3, 7},        // Sub-block 0
		{8, 12, 9, 13, 10, 14, 11, 15},  // Sub-block 1
		{0, 4, 8, 12, 1, 5, 9, 13},      // Sub-block 2 (alternative scan order)
		{2, 6, 10, 14, 3, 7, 11, 15}     // Sub-block 3 (alternative scan order)
	};

	// Pointers to the 4 sub-blocks (each 8 colors). Two from vertical scan, two from horizontal.
	union Color* sub_block_src[4] = {ver_src, ver_src + 8, hor_src, hor_src + 8};

	union Color sub_block_avg[4]; // Stores average colors for each sub-block.
	bool use_differential[2] = {true, true}; // Flags for differential encoding for pairs of sub-blocks.

	// Block Logic: Calculate average colors for pairs of sub-blocks and determine if differential encoding is suitable.
	// Invariant: This loop processes sub-blocks (0,1) and (2,3) to decide on 444 or 555 color format.
	for (uint i = 0, j = 1; i < 4; i += 2, j += 2) {
		float avg_color_0[3];
		getAverageColor(sub_block_src[i], avg_color_0);
		union Color avg_color_555_0 = makeColor555(avg_color_0); // Note: makeColor555 has suspicious logic.

		float avg_color_1[3];
		getAverageColor(sub_block_src[j], avg_color_1);
		union Color avg_color_555_1 = makeColor555(avg_color_1); // Note: makeColor555 has suspicious logic.

		// Determine if the difference between average colors is too large for 5-bit differential encoding.
		for (uint light_idx = 0; light_idx < 3; ++light_idx) {
			int u = avg_color_555_0.components[light_idx] >> 3; // Get 5-bit value (upper 5 bits of uchar).
			int v = avg_color_555_1.components[light_idx] >> 3; // Get 5-bit value (upper 5 bits of uchar).

			int component_diff = v - u;
			// If difference is too large (outside -4 to 3 range for 3-bit differential), use 444 encoding.
			// The condition `component_diff < -3 || component_diff > 3` means `abs(component_diff) > 3`.
			if (component_diff < -3 || component_diff > 3) { // This condition is specific for 3-bit diff.
				use_differential[i / 2] = false; // Disable differential for this pair.
				sub_block_avg[i] = makeColor444(avg_color_0); // Use 444 for first color.
				sub_block_avg[j] = makeColor444(avg_color_1); // Use 444 for second color.
			} else {
				sub_block_avg[i] = avg_color_555_0; // Use 555 for first color.
				sub_block_avg[j] = avg_color_555_1; // Use 555 for second color.
			}
		}
	}

	// Block Logic: Calculate the total color error for each sub-block's average color.
	// This helps determine the best orientation (flip or no flip) for the block.
	uint sub_block_err[4] = {0};
	for (uint i = 0; i < 4; ++i) {
		for (uint j = 0; j < 8; ++j) {
			sub_block_err[i] += getColorError(sub_block_avg[i], sub_block_src[i][j]);
		}
	}

	// Determine if flipping the block (swapping sub-block pairs) reduces total error.
	// Comparing (error of sub-blocks 2+3) with (error of sub-blocks 0+1).
	bool flip =
	sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1];

	// Initialize the destination block to zeros.
	memset(dst, 0, 8);

	// Write the differential and flip flags to the destination block.
	WriteDiff(dst, use_differential[!!flip]); // Use differential flag based on flip decision.
	WriteFlip(dst, flip);

	// Select the correct sub-block average colors based on the flip decision.
	uchar sub_block_off_0 = flip ? 2 : 0;
	uchar sub_block_off_1 = sub_block_off_0 + 1;

	// Write the selected average colors (either 555 differential or 444 packed) to the destination block.
	if (use_differential[!!flip]) {
		WriteColors555(dst, sub_block_avg[sub_block_off_0],
					   sub_block_avg[sub_block_off_1]);
	} else {
		WriteColors444(dst, sub_block_avg[sub_block_off_0],
					   sub_block_avg[sub_block_off_1]);
	}

	return 0; // Return 0, indicating successful compression (error not fully propagated/calculated).
}

/**
 * @brief Copies color data from a source global memory buffer to local Color union array.
 * This helper function extracts 2 `Color` unions (8 bytes total) from the source buffer.
 * @param blocks Pointer to the local array of Color unions to fill.
 * @param index Starting index in the `blocks` array.
 * @param offset Byte offset in the source global buffer.
 * @param src Pointer to the global memory source buffer.
 */
inline void memcpy_colors(union Color* blocks, int index,
			int offset, __global uchar *src)
{
		uchar *values1;
		uchar *values2;

		// Cast the Color unions to uchar pointers to copy byte by byte.
		values1 = (uchar *) &blocks[index];
		values2 = (uchar *) &blocks[index + 1];

		// Copy 4 bytes for the first color and 4 bytes for the second color.
		// Invariant: Copies 8 bytes from `src + offset` into two `Color` unions.
		for (int i = 0; i < 4; i++) {
				values1[i] = *(src+ offset + i);
				values2[i] = *(src+ offset + 4 + i);
		}
}


/**
 * @brief OpenCL kernel for parallel texture compression.
 * Each work-item compresses a 4x4 block of pixels.
 * @param src Global memory pointer to the source uncompressed image data (BGRA).
 * @param dst Global memory pointer to the destination compressed image data.
 * @param width Width of the overall image in pixels.
 * @param height Height of the overall image in pixels.
 */
__kernel void
compression_kernel(__global uchar* src,
		__global uchar* dst,
		int width,
    int height)
{
	// Get global ID for the current work-item (block coordinates).
	int gid_0 = get_global_id(0); // Block X coordinate.
	int gid_1 = get_global_id(1); // Block Y coordinate.

	union Color ver_blocks[16]; // Buffer for 16 colors (4x4 block) scanned vertically.
	union Color hor_blocks[16]; // Buffer for 16 colors (4x4 block) scanned horizontally.

	ulong compressed_error = 0; // Accumulator for compression error.

	// Calculate source offset for the current 4x4 block (4 bytes per pixel, 4 pixels per row, gid_0 blocks per row).
	int src_offset = 4 * 4 * gid_0 + gid_1 * width * 4 * 4;
	// Calculate destination offset for the current 8-byte compressed block.
	int dst_offset = gid_0 * 8 + 8 * gid_1 * width / 4;

	src += src_offset; // Adjust source pointer to the start of the current block.

	// Block Logic: Copy 4x4 pixel data from `src` into `ver_blocks` (vertical scan order) and `hor_blocks` (horizontal scan order).
	// This part seems incomplete/incorrect in the original code, as `x` is incremented but `src` is only adjusted once.
	// It appears to be attempting to load two 4x2 sub-blocks (8 pixels each) for vertical and horizontal passes.
	for (int x = 0; x < width; x += 4) { // This loop structure is suspicious given `src` is fixed and `x` refers to pixel columns.
		// It seems to be trying to load a 4x4 block by reading 4 rows of 4 pixels each.
		// The `memcpy_colors` arguments for `src + x` and `src + x + 8` need careful review for correctness in accessing 4x4 block data.
		memcpy_colors(ver_blocks, 0, 0, src + x); // Likely loads first 2 pixels of row 0
	 	memcpy_colors(ver_blocks, 2, width, src + x); // Likely loads first 2 pixels of row 1 (offset by `width`)
		memcpy_colors(ver_blocks, 4, width * 2, src + x); // Likely loads first 2 pixels of row 2
		memcpy_colors(ver_blocks, 6, width * 3, src + x); // Likely loads first 2 pixels of row 3

		memcpy_colors(hor_blocks, 0, 0, src + x); // Loads first 2 pixels of row 0
		memcpy_colors(hor_blocks, 2, 0, src + x + 8); // Loads next 2 pixels of row 0 (offset by 8 bytes - 2 pixels)
		memcpy_colors(hor_blocks, 4, width, src + x); // Loads first 2 pixels of row 1
		memcpy_colors(hor_blocks, 6, width, src + x + 8); // Loads next 2 pixels of row 1

		memcpy_colors(ver_blocks, 8, 0, src + x + 8); // Likely loads next 2 pixels of row 0
	 	memcpy_colors(ver_blocks, 10, width, src + x + 8); // Likely loads next 2 pixels of row 1
		memcpy_colors(ver_blocks, 12, width * 2, src + x + 8); // Likely loads next 2 pixels of row 2
		memcpy_colors(ver_blocks, 14, width * 3, src + x + 8); // Likely loads next 2 pixels of row 3

		memcpy_colors(hor_blocks, 8, 2 * width, src + x); // Loads first 2 pixels of row 2
	 	memcpy_colors(hor_blocks, 10, 2 * width, src + x + 8); // Loads next 2 pixels of row 2
		memcpy_colors(hor_blocks, 12, 3 * width, src + x); // Loads first 2 pixels of row 3
		memcpy_colors(hor_blocks, 14, 3 * width, src + x + 8); // Loads next 2 pixels of row 3
		break; // The loop breaks after the first iteration, indicating it's processing a single 4x4 block.
	}

	dst += dst_offset; // Adjust destination pointer to the start of the current compressed block.
	// Call the main compression function for the block.
	compressed_error += compressBlock(dst, ver_blocks, hor_blocks, UINT_MAX);

} // End of compression_kernel

// The following code appears to be C++ host-side code for OpenCL setup and execution,
// and it should logically be in a separate .cpp file that includes this .cl file as a string.
// However, as per instructions, it's treated as part of the C++ codebase here.

#include "compress.hpp" // Includes a header for compression-related C++ definitions.

using namespace std; // Use standard namespace for C++.

// Buffer size definitions (constants).
#define BUF_2M		(2 * 1024 * 1024)
#define BUF_32M		(32 * 1024 * 1024)
#define BUF_128	(128)

/**
 * @brief Discovers and selects an OpenCL device (GPU) based on platform and device indices.
 * @param device Reference to a cl_device_id to store the selected device.
 * @param platform_select The index of the desired OpenCL platform.
 * @param device_select The index of the desired OpenCL device within the platform.
 */
void gpu_find(cl_device_id &device,
		uint platform_select,
		uint device_select)
{
	cl_platform_id platform; // Handle for the OpenCL platform.
	cl_uint platform_num = 0; // Number of available platforms.
	cl_platform_id* platform_list = NULL; // List of platform IDs.

	cl_uint device_num = 0; // Number of available devices on a platform.
	cl_device_id* device_list = NULL; // List of device IDs.

	size_t attr_size = 0; // Size of attribute data.
	cl_char* attr_data = NULL; // Buffer for attribute data (e.g., vendor name).

	// Block Logic: Discover all available OpenCL platforms.
	CL_ERR( clGetPlatformIDs(0, NULL, &platform_num)); // Get number of platforms.
	platform_list = new cl_platform_id[platform_num]; // Allocate memory for platform IDs.
	DIE(platform_list == NULL, "alloc platform_list"); // Error check for allocation.

	CL_ERR( clGetPlatformIDs(platform_num, platform_list, NULL)); // Get actual platform IDs.
	cout << "Platforms found: " << platform_num << endl;

	// Block Logic: Iterate through platforms to find devices.
	for(uint platf=0; platf<platform_num; platf++)
	{
		// Get and print platform vendor information.
		CL_ERR( clGetPlatformInfo(platform_list[platf],
				CL_PLATFORM_VENDOR, 0, NULL, &attr_size));
		attr_data = new cl_char[attr_size];
		DIE(attr_data == NULL, "alloc attr_data");

		CL_ERR( clGetPlatformInfo(platform_list[platf],
				CL_PLATFORM_VENDOR, attr_size, attr_data, NULL));
		cout << "Platform " << platf << " " << attr_data << " ";
		delete[] attr_data;

		// Get and print platform version information.
		CL_ERR( clGetPlatformInfo(platform_list[platf],
				CL_PLATFORM_VERSION, 0, NULL, &attr_size));
		attr_data = new cl_char[attr_size];
		DIE(attr_data == NULL, "alloc attr_data");

		CL_ERR( clGetPlatformInfo(platform_list[platf],
				CL_PLATFORM_VERSION, attr_size, attr_data, NULL));
		cout << attr_data << endl;
		delete[] attr_data;

		platform = platform_list[platf]; // Select the current platform.
		DIE(platform == 0, "platform selection"); // Error check.

		// Get number of GPU devices on the current platform.
		if(clGetDeviceIDs(platform,
			CL_DEVICE_TYPE_GPU, 0, NULL, &device_num) == CL_DEVICE_NOT_FOUND) {
			device_num = 0;
			continue; // If no GPU devices, continue to the next platform.
		}

		device_list = new cl_device_id[device_num]; // Allocate memory for device IDs.
		DIE(device_list == NULL, "alloc devices"); // Error check.

		CL_ERR( clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU,
			  device_num, device_list, NULL)); // Get actual device IDs.
		cout << "\tDevices found " << device_num  << endl;

		device = device_list[0]; // Default to the first device.

		// Block Logic: Iterate through devices on the current platform.
		for(uint dev=0; dev<device_num; dev++)
		{
			// Get and print device name.
			CL_ERR( clGetDeviceInfo(device_list[dev], CL_DEVICE_NAME,
				0, NULL, &attr_size));
			attr_data = new cl_char[attr_size];
			DIE(attr_data == NULL, "alloc attr_data");

			CL_ERR( clGetDeviceInfo(device_list[dev], CL_DEVICE_NAME,
				attr_size, attr_data, NULL));
			cout << "\tDevice " << dev << " " << attr_data << " ";
			delete[] attr_data;

			// Get and print device OpenCL version.
			CL_ERR( clGetDeviceInfo(device_list[dev], CL_DEVICE_VERSION,
				0, NULL, &attr_size));
			attr_data = new cl_char[attr_size];
			DIE(attr_data == NULL, "alloc attr_data");

			CL_ERR( clGetDeviceInfo(device_list[dev], CL_DEVICE_VERSION,
				attr_size, attr_data, NULL));
			cout << attr_data;
			delete[] attr_data;

			// If current platform and device match the selection, mark as selected.
			if((platf == platform_select) && (dev == device_select)){
				device = device_list[dev];
				cout << " <--- SELECTED ";
				break; // Break from device loop if selected.
			}

			cout << endl;
		}
	}

	delete[] platform_list; // Clean up allocated memory.
	delete[] device_list; // Clean up allocated memory.
}

/**
 * @brief Constructor for the TextureCompressor class.
 * Initializes OpenCL context and command queue by finding a suitable GPU device.
 */
TextureCompressor::TextureCompressor() {

	int platform_select = 0; // Default platform selection.
	int device_select = 0; // Default device selection.

	gpu_find(device, platform_select, device_select); // Find and select a GPU device.
	DIE(device == 0, "check valid device"); // Error check if no valid device was found.
}

/**
 * @brief Destructor for the TextureCompressor class.
 * (Currently empty, but would typically release OpenCL resources).
 */
TextureCompressor::~TextureCompressor() { }

/**
 * @brief Compresses an image using the OpenCL kernel.
 * This method sets up OpenCL buffers, enqueues the kernel, and reads back the results.
 * @param src Pointer to the source uncompressed image data (uint8_t array).
 * @param dst Pointer to the destination buffer for compressed image data (uint8_t array).
 * @param width Width of the image in pixels.
 * @param height Height of the image in pixels.
 * @return An unsigned long representing the compression result (currently returns 0).
 */
unsigned long TextureCompressor::compress(const uint8_t* src,
									  uint8_t* dst,
									  int width,
									  int height)
{
  cl_int ret; // Variable to store OpenCL API return codes.
  string kernel_src; // String to hold the kernel source code.

  // Block Logic: Create OpenCL context.
  context = clCreateContext(0, 1, &device, NULL, NULL, &ret); // Create context with selected device.
  CL_ERR( ret ); // Error check.

  // Block Logic: Create OpenCL command queue.
  command_queue = clCreateCommandQueue(context, device,
									CL_QUEUE_PROFILING_ENABLE, &ret); // Create command queue with profiling enabled.
  CL_ERR( ret ); // Error check.

  int source_size = 4 * width * height; // Calculate size of source image data (4 bytes per pixel).
  int destination_size = 4 * width * height / 8; // Calculate size of destination compressed data (assuming 8:1 compression for a 4x4 block).

  // Block Logic: Create OpenCL device buffers for source and destination data.
  cl_mem src_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY,
											sizeof(uint8_t) * source_size,
          						NULL, &ret); // Create read-only buffer for source.
  CL_ERR( ret ); // Error check.

  cl_mem dst_buffer = clCreateBuffer(context,	CL_MEM_READ_WRITE,
		 									sizeof(uint8_t) * destination_size,
											NULL, &ret); // Create read-write buffer for destination.
  CL_ERR( ret ); // Error check.

  DIE(src_buffer == 0, "alloc src_buffer"); // Error check for buffer allocation.
  DIE(dst_buffer == 0, "alloc dst_buffer"); // Error check for buffer allocation.

  // Block Logic: Enqueue write operation to transfer source image data from host to device.
  CL_ERR( clEnqueueWriteBuffer(command_queue, src_buffer, CL_TRUE,
						0, sizeof(uint8_t) * source_size, src,
          	0, NULL, NULL)); // Blocking write.

  // Block Logic: Read the kernel source code from file.
  read_kernel("compression_device.cl", kernel_src); // Function to read kernel source (defined elsewhere).
  const char* kernel_c_str = kernel_src.c_str(); // Convert string to C-style string.

  // Block Logic: Create OpenCL program from source.
  program = clCreateProgramWithSource(context, 1, &kernel_c_str, NULL, &ret);
  CL_ERR( ret ); // Error check.

  // Block Logic: Build the OpenCL program (compile the kernel).
  ret = clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", // Build with fast-relaxed-math optimization.
	 			NULL, NULL);
  CL_COMPILE_ERR( ret, program, device ); // Error check for compilation.

  // Block Logic: Create OpenCL kernel object.
  kernel = clCreateKernel(program, "compression_kernel", &ret); // Create kernel from program.
  CL_ERR( ret ); // Error check.

  // Block Logic: Set kernel arguments.
  CL_ERR( clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&src_buffer) ); // Set source buffer argument.
  CL_ERR( clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&dst_buffer) ); // Set destination buffer argument.
  CL_ERR( clSetKernelArg(kernel, 2, sizeof(cl_uint), (void *)&width) );     // Set width argument.
  CL_ERR( clSetKernelArg(kernel, 3, sizeof(cl_uint), (void *)&height) );    // Set height argument.

  // Block Logic: Enqueue the OpenCL kernel for execution.
  cl_event event; // Event for tracking kernel execution.
  size_t globalSize[2] = {(size_t) width / 4, (size_t) height / 4}; // Global work size (number of 4x4 blocks).
  ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, globalSize, 0, 0,
  			NULL, &event); // Enqueue 2D kernel execution.
  CL_ERR( ret ); // Error check.
  CL_ERR( clWaitForEvents(1, &event)); // Wait for kernel execution to complete.

  // Block Logic: Enqueue read operation to transfer compressed data from device to host.
  CL_ERR( clEnqueueReadBuffer(command_queue, dst_buffer, CL_TRUE, 0,
            sizeof(uint8_t) * destination_size, dst, 0, NULL, NULL)); // Blocking read.

  // Block Logic: Finish all enqueued OpenCL commands.
  CL_ERR( clFinish(command_queue) ); // Ensures all commands in queue are complete.

  // Block Logic: Release OpenCL memory objects.
  CL_ERR( clReleaseMemObject(src_buffer) ); // Release source buffer.
  CL_ERR( clReleaseMemObject(dst_buffer) ); // Release destination buffer.

  return 0; // Return 0 (no specific error value returned by this function).
}