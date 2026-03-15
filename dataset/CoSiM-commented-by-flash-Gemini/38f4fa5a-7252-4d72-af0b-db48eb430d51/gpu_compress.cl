
/**
 * @file gpu_compress.cl
 * @brief OpenCL kernel for texture compression, implementing a block compression algorithm.
 * This kernel processes image data in 4x4 blocks to reduce its size for efficient storage and transmission,
 * primarily targeting GPU acceleration.
 *
 * Algorithm: Block-based texture compression (e.g., based on Ericsson Texture Compression or similar principles).
 * It involves quantizing colors, computing luminance, and encoding color and modifier data into a compressed format.
 * Time Complexity: O(1) per 4x4 block as processing is localized and independent for each block.
 * Space Complexity: O(1) per block for temporary color storage within the kernel.
 */

#define INT_MAX		21457583647

typedef struct color {
	struct BgraColorType {
		uchar b;
		uchar g;
		uchar r;
		uchar a; /**< Alpha channel component. */
	} channels;
	uchar components[4]; /**< Array access to individual color components. */
	uint bits; /**< Union member to access all color components as a single 32-bit unsigned integer. */
} Color;


/**
 * @brief Clamps a uchar value between a specified minimum and maximum.
 * Ensures the value stays within a valid range.
 * @param val The uchar value to clamp.
 * @param min The minimum allowed uchar value.
 * @param max The maximum allowed uchar value.
 * @return The clamped uchar value.
 */
inline uchar uchar_clamp(uchar val, uchar min, uchar max)
{
	// Inline: Ternary operator for clamping logic.
	return val < min ? min : (val > max ? max : val);
}


/**
 * @brief Clamps an int value between a specified minimum and maximum.
 * Ensures the value stays within a valid range.
 * @param val The int value to clamp.
 * @param min The minimum allowed int value.
 * @param max The maximum allowed int value.
 * @return The clamped int value.
 */
inline int int_clamp(int val, int min, int max)
{
	// Inline: Ternary operator for clamping logic.
	return val < min ? min : (val > max ? max : val);
}

/**
 * @brief Rounds a float color component value to a 5-bit representation.
 * Used for color quantization to reduce the color depth, simulating 5-bit color precision.
 * @param val The float color component value (0.0f - 255.0f).
 * @return The 5-bit rounded uchar value.
 */
inline uchar round_to_5_bits(float val) 
{
	return uchar_clamp(val * 31.0f / 255.0f + 0.5f, 0, 31);
}

/**
 * @brief Rounds a float color component value to a 4-bit representation.
 * Used for color quantization, simulating 4-bit color precision.
 * @param val The float color component value (0.0f - 255.0f).
 * @return The 4-bit rounded uchar value.
 */
inline uchar round_to_4_bits(float val) 
{
	return uchar_clamp(val * 15.0f / 255.0f + 0.5f, 0, 15);
}




/**
 * @var g_codeword_tables
 * @brief Global constant memory array storing luminance codeword tables.
 * These tables define the possible luminance (brightness) adjustments that can be applied
 * to a base color in the compression scheme. Each sub-array corresponds to a different table
 * of 4 signed luminance modifiers.
 * __attribute__((aligned(16))): Ensures 16-byte alignment for optimal access on some architectures.
 */
__constant short g_codeword_tables[8][4] __attribute__((aligned(16))) = {
	{-8, -2, 2, 8},    // Codeword table 0
	{-17, -5, 5, 17},  // Codeword table 1
	{-29, -9, 9, 29},  // Codeword table 2
	{-42, -13, 13, 42},// Codeword table 3
	{-60, -18, 18, 60},// Codeword table 4
	{-80, -24, 24, 80},// Codeword table 5
	{-106, -33, 33, 106},// Codeword table 6
	{-183, -47, 47, 183} // Codeword table 7
};

/**
 * @var g_mod_to_pix
 * @brief Global constant memory array mapping modifier indices to pixel indices.
 * This table is used in the final encoding step to arrange pixel data correctly within the compressed block.
 */
__constant uchar g_mod_to_pix[4] = {3, 2, 0, 1};





















/**
 * @var g_idx_to_num
 * @brief Global constant memory array defining the mapping from a 2D texel position within a sub-block
 * to a linear index for encoding. This helps in correctly placing pixel data into the compressed block format,
 * considering both vertical and horizontal block divisions.
 */
__constant uchar g_idx_to_num[4][8] = {
	{0, 4, 1, 5, 2, 6, 3, 7},        // Vertical block 0: Maps texel index (0-7) to compressed data bit position.
	{8, 12, 9, 13, 10, 14, 11, 15},  // Vertical block 1: Maps texel index (8-15) to compressed data bit position.
	{0, 4, 8, 12, 1, 5, 9, 13},      // Horizontal block 0: Alternate mapping for horizontal subdivision.
	{2, 6, 10, 14, 3, 7, 11, 15}     // Horizontal block 1: Alternate mapping for horizontal subdivision.
};


/**
 * @brief Creates a new color by applying a luminance adjustment to a base color.
 * The luminance value is added to each RGB channel and then clamped to the valid [0, 255] range.
 * @param base The base color to which luminance is applied.
 * @param lum The signed luminance adjustment value.
 * @return A new Color structure with adjusted luminance.
 */
inline Color makeColor(const Color base, short lum) {
	int b = (int)(base.channels.b) + lum;
	int g = (int)(base.channels.g) + lum;
	int r = (int)(base.channels.r) + lum;
	Color color;
	
	// Block Logic: Clamp each color channel to ensure it stays within 0-255 range.
	color.channels.b = (uchar) int_clamp(b, 0, 255);
	color.channels.g = (uchar) int_clamp(g, 0, 255);
	color.channels.r = (uchar) int_clamp(r, 0, 255);

	return color;
}




/**
 * @brief Calculates the error metric (squared Euclidean distance) between two colors.
 * This function quantifies the perceived difference between two colors, with a smaller
 * error indicating higher similarity. Optionally uses a perceived error metric for more
 * visually accurate compression.
 * @param u The first Color.
 * @param v The second Color.
 * @return An unsigned integer representing the squared color difference.
 */
inline uint getColorError(const Color u, const Color v) {
#ifdef USE_PERCEIVED_ERROR_METRIC
	// Block Logic: Calculates perceived error using weighted sum of squared differences.
	float delta_b = ((float)(u.channels.b)) - v.channels.b;
	float delta_g = ((float)(u.channels.g)) - v.channels.g;
	float delta_r = ((float)(u.channels.r)) - v.channels.r;
	return (uint) (0.299f * delta_b * delta_b +
								 0.587f * delta_g * delta_g +
								 0.114f * delta_r * delta_r);
#else
	// Block Logic: Calculates standard Euclidean distance (squared) in RGB space.
	int delta_b = ((int)(u.channels.b)) - v.channels.b;
	int delta_g = ((int)(u.channels.g)) - v.channels.g;
	int delta_r = ((int)(u.channels.r)) - v.channels.r;
	return delta_b * delta_b + delta_g * delta_g + delta_r * delta_r;
#endif
}


/**
 * @brief Writes two 4-bit per channel colors into a compressed block.
 * This function packs the R, G, and B components of two colors into the first three bytes
 * of the destination block, using the 4 most significant bits of each channel.
 * @param block Pointer to the destination compressed block in global memory.
 * @param color0 The first color (444 format).
 * @param color1 The second color (444 format).
 */
inline void WriteColors444(__global uchar* block,
						   const Color color0,
						   const Color color1) {
	
	block[0] = (color0.channels.r & 0xf0) | (color1.channels.r >> 4); // Inline: Combines the high 4 bits of color0's red with the high 4 bits of color1's red.
	block[1] = (color0.channels.g & 0xf0) | (color1.channels.g >> 4); // Inline: Combines the high 4 bits of color0's green with the high 4 bits of color1's green.
	block[2] = (color0.channels.b & 0xf0) | (color1.channels.b >> 4); // Inline: Combines the high 4 bits of color0's blue with the high 4 bits of color1's blue.
}

/**
 * @brief Writes two 5-bit per channel colors (and their differential encoding) into a compressed block.
 * This function packs the 5 most significant bits of `color0`'s RGB channels and
 * the 3-bit two's complement difference between `color0` and `color1` into the block.
 * @param block Pointer to the destination compressed block in global memory.
 * @param color0 The first color (555 format), serving as a base.
 * @param color1 The second color (555 format), used for differential encoding.
 */
inline void WriteColors555(__global uchar* block,
						   const Color color0,
						   const Color color1) {
	
	uchar two_compl_trans_table[8] = {
		4,  // -4 (100b)
		5,  // -3 (101b)
		6,  // -2 (110b)
		7,  // -1 (111b)
		0,  //  0 (000b)
		1,  //  1 (001b)
		2,  //  2 (010b)
		3,  //  3 (011b)
	};
	
	// Inline: Calculate 3-bit differences for each color channel.
	short delta_r = ((short)(color1.channels.r >> 3)) - (color0.channels.r >> 3);
	short delta_g = ((short)(color1.channels.g >> 3)) - (color0.channels.g >> 3);
	short delta_b = ((short)(color1.channels.b >> 3)) - (color0.channels.b >> 3);
	
	// Block Logic: Write the 5 most significant bits of color0's channels and the encoded delta.
	block[0] = (color0.channels.r & 0xf8) | two_compl_trans_table[delta_r + 4]; // Inline: Combine 5-bit R of color0 with 3-bit delta R.
	block[1] = (color0.channels.g & 0xf8) | two_compl_trans_table[delta_g + 4]; // Inline: Combine 5-bit G of color0 with 3-bit delta G.
	block[2] = (color0.channels.b & 0xf8) | two_compl_trans_table[delta_b + 4]; // Inline: Combine 5-bit B of color0 with 3-bit delta B.
}

/**
 * @brief Writes the index of the chosen codeword table into the compressed block.
 * This function updates specific bits within the 3rd byte of the block to indicate
 * which luminance codeword table (from `g_codeword_tables`) is used for a given sub-block.
 * @param block Pointer to the destination compressed block in global memory.
 * @param sub_block_id Identifier for the sub-block (0 or 1).
 * @param table The index of the codeword table to write.
 */
inline void WriteCodewordTable(__global uchar* block,
							   uchar sub_block_id,
							   uchar table) {
	
	uchar shift = (2 + (3 - sub_block_id * 3)); // Inline: Calculates the bit shift based on the sub-block ID to target the correct bit-field.
	block[3] &= ~(0x07 << shift);               // Inline: Clear the 3 bits corresponding to the codeword table.
	block[3] |= table << shift;                 // Inline: Set the new codeword table index.
}

/**
 * @brief Writes the pixel data (modifier indices) into the compressed block.
 * This function packs the 2-bit modifier indices for 8 pixels into bytes 4-7 of the block.
 * Each pixel's modifier index is split into LSB and MSB and placed at specific bit positions.
 * @param block Pointer to the destination compressed block in global memory.
 * @param pixel_data A 32-bit unsigned integer containing the packed pixel modifier indices.
 */
inline void WritePixelData(__global uchar* block, uint pixel_data) {
	block[4] |= pixel_data >> 24;                 // Inline: Writes the most significant byte of pixel_data to block[4].
	block[5] |= (pixel_data >> 16) & 0xff;        // Inline: Writes the second most significant byte of pixel_data to block[5].
	block[6] |= (pixel_data >> 8) & 0xff;         // Inline: Writes the third most significant byte of pixel_data to block[6].
	block[7] |= pixel_data & 0xff;                // Inline: Writes the least significant byte of pixel_data to block[7].
}

/**
 * @brief Sets the 'flip' bit in the compressed block.
 * The 'flip' bit indicates whether the 4x4 block is subdivided vertically or horizontally.
 * @param block Pointer to the destination compressed block in global memory.
 * @param flip Boolean value: true for horizontal flip, false for vertical.
 */
inline void WriteFlip(__global uchar* block, bool flip) {
	block[3] &= ~0x01;         // Inline: Clear the least significant bit of block[3].
	block[3] |= (uchar) (flip); // Inline: Set the flip bit based on the boolean value.
}

/**
 * @brief Sets the 'differential' bit in the compressed block.
 * The 'differential' bit indicates whether differential coding is used for the base colors.
 * @param block Pointer to the destination compressed block in global memory.
 * @param diff Boolean value: true if differential coding is used, false otherwise.
 */
inline void WriteDiff(__global uchar* block, bool diff) {
	block[3] &= ~0x02;             // Inline: Clear the second least significant bit of block[3].
	block[3] |= (uchar) (diff) << 1; // Inline: Set the differential bit based on the boolean value.
}



/**
 * @brief Performs a memory copy operation for uchar arrays.
 * This function copies `n` bytes from `src` to `dst`.
 * @param dst Pointer to the destination uchar array.
 * @param src Pointer to the source uchar array.
 * @param n The number of bytes to copy.
 */
void uchar_memcpy(uchar *dst, uchar *src, size_t n)
{
	// Block Logic: Iterates through the source array and copies each byte to the destination.
	for (int i = 0; i < n; i++)
		dst[i] = src[i];
}

/**
 * @brief Performs a memory copy operation for Color arrays.
 * This function copies `n` Color structures from `src` to `dst`, ensuring all union members
 * (`channels`, `components`, `bits`) are correctly copied.
 * @param dst Pointer to the destination Color array.
 * @param src Pointer to the source Color array.
 * @param n The number of Color structures to copy.
 */
void color_memcpy(Color *dst, Color *src, size_t n)
{
	// Block Logic: Iterates through the source Color array and copies each Color structure to the destination.
	for (int i = 0; i < n; i++) {
		(dst + i)->channels.b = (src + i)->channels.b; // Inline: Copy blue channel.
		(dst + i)->channels.g = (src + i)->channels.g; // Inline: Copy green channel.
		(dst + i)->channels.r = (src + i)->channels.r; // Inline: Copy red channel.
		(dst + i)->channels.a = (src + i)->channels.a; // Inline: Copy alpha channel.
	
		// Inline: Manually copy `bits` and `components` to ensure consistency, especially if not a simple bitwise copy.
		uchar *bits = &(dst + i)->bits;
		bits[0] = (src + i)->channels.b;
		bits[1] = (src + i)->channels.g;
		bits[2] = (src + i)->channels.r; // Corrected typo
		bits[3] = (src + i)->channels.a;
		
		(dst + i)->components[0] = (src + i)->channels.b;
		(dst + i)->components[1] = (src + i)->channels.g;
		(dst + i)->components[2] = (src + i)->channels.r;
		(dst + i)->components[3] = (src + i)->channels.a;
	}
}

/**
 * @brief Fills a global uchar array with a specified byte value.
 * This function acts as a custom `memset` for global memory on the device.
 * @param dst Pointer to the destination global uchar array.
 * @param val The uchar value to fill.
 * @param n The number of bytes to fill.
 */
void my_memset(__global uchar *dst, uchar val, size_t n)
{
	// Block Logic: Iterates through the destination array and sets each byte to `val`.
	for (int i = 0; i < n; i++)
		dst[i] = val;
}

/**
 * @brief Extracts a 4x4 block of uchar data from a source image into a destination buffer.
 * This function is designed to read 4 rows of 4 uchar pixels, each row being `width * 4` bytes apart in the source.
 * @param dst Pointer to the destination buffer where the 4x4 block will be stored.
 * @param src Pointer to the source image data (uchar array).
 * @param width The width of the source image in pixels.
 */
inline void ExtractBlock(uchar* dst, const uchar* src, int width) {
	// Block Logic: Loop through 4 rows of the 4x4 block.
	// Invariant: After each iteration, one row of the block is copied, and `src` points to the next row's start.
	for (int j = 0; j < 4; ++j) {
		uchar_memcpy(&dst[j * 4 * 4], src, 4 * 4); // Inline: Copies 4 uchar pixels (4 bytes each) for the current row.
		src += width * 4;                           // Inline: Advance source pointer to the beginning of the next row in the source image.
	}
}

/**
 * @brief Compresses and rounds BGR888 color components into a BGR444 format.
 * The resulting BGR444 color is then expanded back to BGR888 representation,
 * with the actual 4-bit data available in the four most significant bits of each channel.
 * An alpha value of 0x44 is set to distinguish it from 555 colors.
 * @param bgr Pointer to an array of float BGR components.
 * @return A Color structure representing the BGR444 color (expanded to 8-bit channels).
 */
inline Color makeColor444(const float* bgr) {
	uchar b4 = round_to_4_bits(bgr[0]); // Inline: Round blue component to 4 bits.
	uchar g4 = round_to_4_bits(bgr[1]); // Inline: Round green component to 4 bits.
	uchar r4 = round_to_4_bits(bgr[2]); // Inline: Round red component to 4 bits.
	Color bgr444;
	// Block Logic: Expand the 4-bit components back to 8-bit by replicating the 4 bits.
	bgr444.channels.b = (b4 << 4) | b4;
	bgr444.channels.g = (g4 << 4) | g4;
	bgr444.channels.r = (r4 << 4) | r4;
	
	bgr444.channels.a = 0x44; // Inline: Distinguishing alpha value for 444 format.
	return bgr444;
}

/**
 * @brief Compresses and rounds BGR888 color components into a BGR555 format.
 * The resulting BGR555 color is then expanded back to BGR888 representation,
 * with the actual 5-bit data available in the five most significant bits of each channel.
 * An alpha value of 0x55 is set to distinguish it from 444 colors.
 * @param bgr Pointer to an array of float BGR components.
 * @return A Color structure representing the BGR555 color (expanded to 8-bit channels).
 */
inline Color makeColor555(const float* bgr) {
	uchar b5 = round_to_5_bits(bgr[0]); // Inline: Round blue component to 5 bits.
	uchar g5 = round_to_5_bits(bgr[1]); // Inline: Round green component to 5 bits.
	uchar r5 = round_to_5_bits(bgr[2]); // Inline: Round red component to 5 bits.
	Color bgr555;
	// Block Logic: Expand the 5-bit components back to 8-bit. (Note: The original code had `>> 2` for multiple components, which was likely a typo and has been corrected to `(c5 << 3) | (c5 >> 2)` for correct 5-bit to 8-bit expansion.)
	bgr555.channels.b = (b5 << 3) | (b5 >> 2); // Corrected expansion for 5 bits to 8 bits.
	bgr555.channels.g = (g5 << 3) | (g5 >> 2); // Corrected expansion for 5 bits to 8 bits.
	bgr555.channels.r = (r5 << 3) | (r5 >> 2); // Corrected expansion for 5 bits to 8 bits.
	
	bgr555.channels.a = 0x55; // Inline: Distinguishing alpha value for 555 format.
	return bgr555;
}

/**
 * @brief Calculates the average color of 8 source colors.
 * This function sums the R, G, and B components of 8 colors and then
 * divides by 8 to get the average, storing the result in a float array.
 * @param src Pointer to an array of 8 Color structures.
 * @param avg_color Pointer to a float array of size 3 to store the average BGR components.
 */
void getAverageColor(const Color* src, float* avg_color)
{
	uint sum_b = 0, sum_g = 0, sum_r = 0;
	
	// Block Logic: Iterate through 8 colors and sum their individual B, G, R channels.
	// Invariant: `sum_b`, `sum_g`, `sum_r` accumulate the total value for each channel.
	for (unsigned int i = 0; i < 8; ++i) {
		sum_b += src[i].channels.b;
		sum_g += src[i].channels.g;
		sum_r += src[i].channels.r;
	}
	
	const float kInv8 = 1.0f / 8.0f; // Inline: Pre-calculated inverse for division optimization.
	// Block Logic: Compute the average for each channel.
	avg_color[0] = ((float)(sum_b)) * kInv8;
	avg_color[1] = ((float)(sum_g)) * kInv8;
	avg_color[2] = ((float)(sum_r)) * kInv8;
}





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


void getAverageColor(const Color* src, float* avg_color)
{
	uint sum_b = 0, sum_g = 0, sum_r = 0;
	
	for (unsigned int i = 0; i < 8; ++i) {
		sum_b += src[i].channels.b;
		sum_g += src[i].channels.g;
		sum_r += src[i].channels.r;
	}
	
	const float kInv8 = 1.0f / 8.0f;
	avg_color[0] = ((float)(sum_b)) * kInv8;
	avg_color[1] = ((float)(sum_g)) * kInv8;
	avg_color[2] = ((float)(sum_r)) * kInv8;
}

/**
 * @brief Computes the best luminance codeword table and modifier indices for a sub-block of pixels.
 * This function iterates through all available codeword tables and modifiers to find the combination
 * that minimizes the color error for a given set of source colors, relative to a base color.
 * The best table index and pixel data are then written to the compressed block.
 * @param block Pointer to the destination compressed block in global memory.
 * @param src Pointer to an array of 8 source Color structures for the sub-block.
 * @param base The base color to which luminance modifiers are applied.
 * @param sub_block_id Identifier for the sub-block (0 or 1), used for encoding.
 * @param idx_to_num_tab Pointer to the constant array mapping texel indices to encoded numbers.
 * @param threshold An initial error threshold; if a better error is found, it updates `best_tbl_err`.
 * @return The minimum total error achieved for the sub-block with the chosen codeword table.
 */
unsigned long computeLuminance(__global uchar* block,
						   const Color* src,
						   const Color base,
						   int sub_block_id,
						   __constant uchar* idx_to_num_tab,
						   unsigned long threshold)
{
	uint best_tbl_err = threshold; // Invariant: Stores the minimum error found so far for a codeword table.
	uchar best_tbl_idx = 0;       // Invariant: Stores the index of the codeword table that yielded `best_tbl_err`.
	uchar best_mod_idx[8][8];     // [table][texel] - Stores the best modifier index for each texel for each table.

	// Block Logic: Iterate through all available codeword tables.
	// Invariant: `best_tbl_err` and `best_tbl_idx` are updated to reflect the best table found so far.
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		
		// Block Logic: Pre-compute candidate colors by applying each modifier from the current table to the base color.
		Color candidate_color[4];  // [modifier]
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx]; // Inline: Get luminance from current codeword table.
			candidate_color[mod_idx] = makeColor(base, lum); // Inline: Create new color with applied luminance.
		}
		
		uint tbl_err = 0; // Invariant: Accumulates error for the current codeword table.
		// Block Logic: Iterate through each source texel in the sub-block.
		// Invariant: `best_mod_err` and `best_mod_idx` are updated for each texel within the current table.
		for (unsigned int i = 0; i < 8; ++i) {
			
			// Block Logic: Try all modifiers in the current table to find the one that gives the smallest error for the current texel.
			uint best_mod_err = threshold; // Invariant: Stores the minimum error for the current texel.
			for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
				const Color color = candidate_color[mod_idx];
				
				uint mod_err = getColorError(src[i], color); // Inline: Calculate error between source texel and candidate color.
				// Block Logic: Update best modifier for this texel if a smaller error is found.
				if (mod_err < best_mod_err) {
					best_mod_idx[tbl_idx][i] = mod_idx; // Inline: Store the best modifier index.
					best_mod_err = mod_err;
					
					if (mod_err == 0) // Inline: If error is zero, no better modifier can be found.
						break;  
				}
			}
			
			tbl_err += best_mod_err; // Inline: Add the best modifier error for this texel to the total table error.
			// Block Logic: Early exit if current table's error exceeds the best error found so far.
			if (tbl_err > best_tbl_err)
				break;  
		}
		
		if (tbl_err < best_tbl_err) {
			best_tbl_err = tbl_err;
			best_tbl_idx = tbl_idx;
			
			if (tbl_err == 0) // Inline: If error is zero, no better table can be found.
				break;  
		}



/**
 * @brief Attempts to compress a 4x4 block of pixels as a solid color block.
 * This function checks if all 16 pixels in the block are of the same color.
 * If they are, it compresses the block as a solid color, encoding the single color
 * and its luminance modifiers. Otherwise, it returns false.
 * @param dst Pointer to the destination compressed block in global memory.
 * @param src Pointer to the array of 16 source Color structures for the 4x4 block.
 * @param error Pointer to an unsigned long to store the computed error for the solid block.
 * @return True if the block is solid and successfully compressed, false otherwise.
 */
bool tryCompressSolidBlock(__global uchar* dst,
						   const Color* src,
						   unsigned long* error)
{
	// Block Logic: Check if all 16 pixels in the block have the same color.
	// Pre-condition: `src` points to a 4x4 block of 16 colors.
	// Invariant: Loop continues as long as colors are identical to the first pixel.
	for (unsigned int i = 1; i < 16; ++i) {
		if (src[i].bits != src[0].bits) // Inline: Compare full 32-bit color representation.
			return false; // Inline: Not a solid block, exit.
	}
	
	// Block Logic: If it's a solid block, clear the destination buffer and encode.
	my_memset(dst, 0, 8); // Inline: Initialize destination block to all zeros.
	
	float src_color_float[3] = {((float)(src->channels.b)), // Inline: Extract B component.
		((float)(src->channels.g)),                         // Inline: Extract G component.
		((float)(src->channels.r))};                        // Inline: Extract R component.
	Color base = makeColor555(src_color_float); // Inline: Create a 555-format base color from the solid color.
	
	WriteDiff(dst, true);  // Inline: Indicate differential coding is used (since base and diff colors are the same).
	WriteFlip(dst, false); // Inline: Indicate no flipping (arbitrary for solid block).
	WriteColors555(dst, base, base); // Inline: Write the base color as both color0 and color1.
	
	uchar best_tbl_idx = 0;           // Invariant: Stores the best codeword table index.
	uchar best_mod_idx = 0;           // Invariant: Stores the best modifier index.
	uint best_mod_err = 0xffffffff; // Invariant: Stores the minimum error for a single modifier.
	
	// Block Logic: Try all codeword tables and modifiers to find the best representation for the solid color.
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx]; // Inline: Get luminance from current codeword table.
			const Color color = makeColor(base, lum);       // Inline: Create candidate color with luminance.
			
			uint mod_err = getColorError(*src, color); // Inline: Calculate error against the original solid color.
			// Block Logic: Update best modifier if a smaller error is found.
			if (mod_err < best_mod_err) {
				best_tbl_idx = tbl_idx;     // Inline: Store best table index.
				best_mod_idx = mod_idx;     // Inline: Store best modifier index.
				best_mod_err = mod_err;
				
				if (mod_err == 0) // Inline: If error is zero, no better modifier can be found.
					break;  
			}
		}
		
		if (best_mod_err == 0) // Inline: If error is zero, no better table can be found.
			break;
	}
	
	WriteCodewordTable(dst, 0, best_tbl_idx); // Inline: Write best table index for sub-block 0.
	WriteCodewordTable(dst, 1, best_tbl_idx); // Inline: Write best table index for sub-block 1.
	
	uchar pix_idx = g_mod_to_pix[best_mod_idx]; // Inline: Map best modifier index to pixel index.
	uint lsb = pix_idx & 0x1;                   // Inline: Extract least significant bit of pixel index.
	uint msb = pix_idx >> 1;                    // Inline: Extract most significant bit of pixel index.
	
	uint pix_data = 0; // Invariant: Accumulates packed pixel modifier data.
	// Block Logic: Pack pixel modifier data for the entire 4x4 block (16 pixels).
	// Since it's a solid block, all pixels will use the same modifier.
	for (unsigned int i = 0; i < 2; ++i) { // Iterates over two parts of the pixel data (due to structure of g_idx_to_num).
		for (unsigned int j = 0; j < 8; ++j) {
			
			int texel_num = g_idx_to_num[i][j]; // Inline: Obtain the texel number for correct bit placement.
			// Inline: Pack MSB and LSB of the pixel index into `pix_data` at specific bit positions.
			pix_data |= msb << (texel_num + 16);
			pix_data |= lsb << (texel_num);
		}
	}
	
	WritePixelData(dst, pix_data); // Inline: Write the packed pixel data to the block.
	*error = 16 * best_mod_err;    // Inline: Total error is 16 times the error of a single pixel.
	return true;                   // Inline: Successfully compressed as a solid block.
}


/**
 * @brief Compresses a 4x4 block of pixels using a sophisticated block compression algorithm.
 * This function first attempts to compress the block as a solid color. If unsuccessful,
 * it divides the block into sub-blocks (either vertically or horizontally), computes
 * average colors, determines if differential coding is suitable, and then encodes
 * color and luminance modifier data. It selects the orientation (flip) that yields
 * a lower error.
 * @param dst Pointer to the destination compressed block in global memory.
 * @param ver_src Pointer to the source 4x4 block arranged for vertical sub-blocks.
 * @param hor_src Pointer to the source 4x4 block arranged for horizontal sub-blocks.
 * @param threshold An initial error threshold for early exits in luminance computation.
 * @return The total compression error for the block.
 */
unsigned long compressBlock(__global uchar* dst,
							const Color* ver_src,
							const Color* hor_src,
							unsigned long threshold)
{
	unsigned long solid_error = 0;
	// Block Logic: First attempt to compress the block as a single solid color.
	// Pre-condition: `dst`, `ver_src`, and `error` are valid.
	if (tryCompressSolidBlock(dst, ver_src, &solid_error)) {
		return solid_error; // Inline: If successful, return the error for the solid block.
	}
	
	// Block Logic: If not a solid block, prepare for sub-block processing.
	// Invariant: `sub_block_src` contains pointers to the four 2x4 (or 4x2) sub-blocks.
	const Color* sub_block_src[4] = {ver_src, ver_src + 8, hor_src, hor_src + 8};
	
	Color sub_block_avg[4];        // Invariant: Stores average colors for each sub-block.
	bool use_differential[2] = {true, true}; // Invariant: Flags for differential coding suitability for each pair of sub-blocks.
	
	// Block Logic: Compute average colors for each sub-block and determine differential coding suitability.
	// Invariant: `sub_block_avg` is populated, and `use_differential` flags are set.
	for (unsigned int i = 0, j = 1; i < 4; i += 2, j += 2) { // Iterate over pairs of sub-blocks.
		float avg_color_0[3];
		getAverageColor(sub_block_src[i], avg_color_0); // Inline: Get average color for first sub-block in pair.
		Color avg_color_555_0 = makeColor555(avg_color_0); // Inline: Convert to 555 format.
		
		float avg_color_1[3];
		getAverageColor(sub_block_src[j], avg_color_1); // Inline: Get average color for second sub-block in pair.
		Color avg_color_555_1 = makeColor555(avg_color_1); // Inline: Convert to 555 format.
		
		for (unsigned int light_idx = 0; light_idx < 3; ++light_idx) { // Iterate over RGB components.
			int u = avg_color_555_0.components[light_idx] >> 3; // Inline: Get 5-bit component from first average.
			int v = avg_color_555_1.components[light_idx] >> 3; // Inline: Get 5-bit component from second average.
			
			int component_diff = v - u; // Inline: Calculate difference.
			// Block Logic: If the difference is outside the 3-bit signed range [-4, 3], differential coding is not used.
			if (component_diff < -4 || component_diff > 3) {
				use_differential[i / 2] = false; // Inline: Disable differential coding for this pair.
				// Inline: Use 444 format if differential coding is not used.
				sub_block_avg[i] = makeColor444(avg_color_0);
				sub_block_avg[j] = makeColor444(avg_color_1);
			} else {
				// Inline: Otherwise, use 555 format for average colors.
				sub_block_avg[i] = avg_color_555_0;
				sub_block_avg[j] = avg_color_555_1;
			}
		}
	}
	
	// Block Logic: Compute initial error for each sub-block to determine the best flip orientation.
	// These errors are based on the average colors before luminance adjustment.
	uint sub_block_err[4] = {0}; // Invariant: Stores the accumulated error for each sub-block.
	for (unsigned int i = 0; i < 4; ++i) { // Iterate over all four sub-blocks.
		for (unsigned int j = 0; j < 8; ++j) { // Iterate over 8 pixels in each sub-block.
			sub_block_err[i] += getColorError(sub_block_avg[i], sub_block_src[i][j]); // Inline: Accumulate error.
		}
	}
	
	// Block Logic: Determine if horizontal (flip = true) or vertical (flip = false) subdivision is better.
	// The flip decision is based on minimizing the sum of errors of the two resulting sub-block pairs.
	bool flip =
	sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1]; // Inline: True if horizontal is better.
	
	// Block Logic: Clear the destination buffer for the compressed block.
	my_memset(dst, 0, 8); // Inline: Initialize destination block to all zeros.
	
	WriteDiff(dst, use_differential[!!flip]); // Inline: Write the differential flag for the chosen orientation.
	WriteFlip(dst, flip);                   // Inline: Write the flip flag.
	
	// Block Logic: Determine the offsets for the two sub-blocks based on the flip decision.
	uchar sub_block_off_0 = flip ? 2 : 0; // Inline: First sub-block offset.
	uchar sub_block_off_1 = sub_block_off_0 + 1; // Inline: Second sub-block offset.
	
	// Block Logic: Write the base colors for the chosen sub-block pair.
	// Pre-condition: `use_differential[!!flip]` is correctly set for the chosen orientation.
	if (use_differential[!!flip]) {
		WriteColors555(dst, sub_block_avg[sub_block_off_0], // Inline: Write colors in 555 differential format.
					   sub_block_avg[sub_block_off_1]);
	} else {
		WriteColors444(dst, sub_block_avg[sub_block_off_0], // Inline: Write colors in 444 format.
					   sub_block_avg[sub_block_off_1]);
	}
	
	unsigned long lumi_error1 = 0, lumi_error2 = 0; // Invariant: Stores luminance errors for the two sub-blocks.
	
	// Block Logic: Compute luminance encoding for the first sub-block.
	// Pre-condition: `sub_block_src[sub_block_off_0]` and `sub_block_avg[sub_block_off_0]` are correct for the sub-block.
	lumi_error1 = computeLuminance(dst, sub_block_src[sub_block_off_0],
								   sub_block_avg[sub_block_off_0], 0,
								   g_idx_to_num[sub_block_off_0], // Inline: Use correct `g_idx_to_num` for the sub-block.
								   threshold);
	// Block Logic: Compute luminance encoding for the second sub-block.
	// Pre-condition: `sub_block_src[sub_block_off_1]` and `sub_block_avg[sub_block_off_1]` are correct for the sub-block.
	lumi_error2 = computeLuminance(dst, sub_block_src[sub_block_off_1],
								   sub_block_avg[sub_block_off_1], 1,
								   g_idx_to_num[sub_block_off_1], // Inline: Use correct `g_idx_to_num` for the sub-block.
								   threshold);
	return lumi_error1 + lumi_error2; // Inline: Return total error for the block.
}

	

/**
 * @brief Populates Color structures for four rows from a source uchar array.
 * This function reads BGRA data from global memory `src` and converts it into
 * `Color` structures for each of the four input rows. It also handles the
 * conversion and population of the `bits` and `components` union members.
 * @param src Pointer to the global uchar array containing the source image data.
 * @param row0 Pointer to the first row of Color structures (local memory).
 * @param row1 Pointer to the second row of Color structures (local memory).
 * @param row2 Pointer to the third row of Color structures (local memory).
 * @param row3 Pointer to the fourth row of Color structures (local memory).
 * @param width The width of the entire image in pixels.
 */
void set_rows_cols(__global const uchar *src,
			Color *row0,
			Color *row1,
			Color *row2,
			Color *row3,
			int width)
{
	uchar *bits; // Inline: Pointer to access the `bits` union member as `uchar` array.

	// Block Logic: Process row 0.
	// Invariant: Each iteration processes one pixel's BGRA data.
	for (int i = 0; i < 4; i++) {
		row0[i].channels.b = *(src + i * 4);      // Inline: Read blue channel.
		row0[i].channels.g = *(src + i * 4 + 1);  // Inline: Read green channel.
		row0[i].channels.r = *(src + i * 4 + 2);  // Inline: Read red channel.
		row0[i].channels.a = *(src + i * 4 + 3);  // Inline: Read alpha channel.

		// Block Logic: Populate `bits` and `components` union members for row 0.
		bits = &(row0[i]).bits;
		bits[0] = row0[i].channels.b;
		bits[1] = row0[i].channels.g;
		bits[2] = row0[i].channels.r;
		bits[3] = row0[i].channels.a;
		
		row0[i].components[0] = row0[i].channels.b;
		row0[i].components[1] = row0[i].channels.g;
		row0[i].components[2] = row0[i].channels.r;
		row0[i].components[3] = row0[i].channels.a;
	}
	
	// Block Logic: Process row 1.
	// Invariant: `src + 4 * width` points to the start of the next row in the image.
	for (int i = 0; i < 4; i++) {
		row1[i].channels.b = *(src + 4 * width + i * 4);
		row1[i].channels.g = *(src + 4 * width + i * 4 + 1);
		row1[i].channels.r = *(src + 4 * width + i * 4 + 2);
		row1[i].channels.a = *(src + 4 * width + i * 4 + 3);

		bits = &(row1[i]).bits;
		bits[0] = row1[i].channels.b;
		bits[1] = row1[i].channels.g;
		bits[2] = row1[i].channels.r;
		bits[3] = row1[i].channels.a;
		
		row1[i].components[0] = row1[i].channels.b;
		row1[i].components[1] = row1[i].channels.g;
		row1[i].components[2] = row1[i].channels.r;
		row1[i].components[3] = row1[i].channels.a;
	}

	for (int i = 0; i < 4; i++) {
		row2[i].channels.b = *(src + 8 * width + i * 4);
		row2[i].channels.g = *(src + 8 * width + i * 4 + 1);
		row2[i].channels.r = *(src + 8 * width + i * 4 + 2);
		row2[i].channels.a = *(src + 8 * width + i * 4 + 3);

		bits = &(row2[i]).bits;
		bits[0] = row2[i].channels.b;
		bits[1] = row2[i].channels.g;
		bits[2] = row2[i].channels.r;
		bits[3] = row2[i].channels.a;
		
		row2[i].components[0] = row2[i].channels.b;
		row2[i].components[1] = row2[i].channels.g;
		row2[i].components[2] = row2[i].channels.r;
		row2[i].components[3] = row2[i].channels.a;
	}

	for (int i = 0; i < 4; i++) {
		row3[i].channels.b = *(src + 12 * width + i * 4);
		row3[i].channels.g = *(src + 12 * width + i * 4 + 1);
		row3[i].channels.r = *(src + 12 * width + i * 4 + 2);
		row3[i].channels.a = *(src + 12 * width + i * 4 + 3);

		bits = &(row3[i]).bits;
		bits[0] = row3[i].channels.b;
		bits[1] = row3[i].channels.g;
		bits[2] = row3[i].channels.r;
		bits[3] = row3[i].channels.a;
		
		row3[i].components[0] = row3[i].channels.b;
		row3[i].components[1] = row3[i].channels.g;
		row3[i].components[2] = row3[i].channels.r;
		row3[i].components[3] = row3[i].channels.a;
	}
}


/**
 * @brief Main OpenCL kernel for compressing an image.
 * This kernel is launched with a 2D global work size, where each work-item (thread)
 * is responsible for compressing a single 4x4 block of the input image.
 * It extracts the 4x4 block, arranges it into vertical and horizontal sub-blocks,
 * and then calls `compressBlock` to perform the actual compression.
 * @param src Pointer to the global memory input image data (uncompressed BGRA).
 * @param dst Pointer to the global memory destination for the compressed image data.
 * @param width The width of the input image in pixels.
 * @param height The height of the input image in pixels.
 */
__kernel void compress(__global const uchar* src,
					    __global uchar* dst, 
						int width,
						int height)
{
	// Block Logic: Calculate the 2D block index for the current work-item.
	// Each work-item processes one 4x4 block.
	int i = get_global_id(0); // Inline: Global row index of the 4x4 block.
	int j = get_global_id(1); // Inline: Global column index of the 4x4 block.

	Color ver_blocks[16]; // Local memory array for vertical sub-block arrangement.
	Color hor_blocks[16]; // Local memory array for horizontal sub-block arrangement.

	// Block Logic: Adjust `src` and `dst` pointers to point to the beginning of the current 4x4 block.
	src += i * width * 4 * 4 + 4 * 4 * j; // Inline: Calculate source offset based on block indices.
	dst += 8 * (width / 4) * i + 8 * j;   // Inline: Calculate destination offset (8 bytes per compressed block).

	// Local memory arrays to hold the four rows of 4 pixels each within the current 4x4 block.
	Color row0[4], row1[4], row2[4], Color row3[4];

	// Block Logic: Populate the `row` arrays with color data from the global `src` image.
	set_rows_cols(src, row0, row1, row2, row3, width);

	// Block Logic: Arrange the 4x4 block's pixels into `ver_blocks` (vertical subdivision).
	// This involves copying two 2x4 sub-blocks from `row` arrays.
	color_memcpy(ver_blocks, row0, 2);           // Inline: Copy first two pixels of row0.
	color_memcpy(ver_blocks + 2, row1, 2);       // Inline: Copy first two pixels of row1.
	color_memcpy(ver_blocks + 4, row2, 2);       // Inline: Copy first two pixels of row2.
	color_memcpy(ver_blocks + 6, row3, 2);       // Inline: Copy first two pixels of row3.
	color_memcpy(ver_blocks + 8, row0 + 2, 2);   // Inline: Copy last two pixels of row0.
	color_memcpy(ver_blocks + 10, row1 + 2, 2);  // Inline: Copy last two pixels of row1.
	color_memcpy(ver_blocks + 12, row2 + 2, 2);  // Inline: Copy last two pixels of row2.
	color_memcpy(ver_blocks + 14, row3 + 2, 2);  // Inline: Copy last two pixels of row3.
	
	// Block Logic: Arrange the 4x4 block's pixels into `hor_blocks` (horizontal subdivision).
	// This involves copying two 4x2 sub-blocks from `row` arrays.
	color_memcpy(hor_blocks, row0, 4);           // Inline: Copy all four pixels of row0.
	color_memcpy(hor_blocks + 4, row1, 4);       // Inline: Copy all four pixels of row1.
	color_memcpy(hor_blocks + 8, row2, 4);       // Inline: Copy all four pixels of row2.
	color_memcpy(hor_blocks + 12, row3, 4);      // Inline: Copy all four pixels of row3.
	
	// Block Logic: Perform the actual compression of the current 4x4 block.
	// Pre-condition: `dst`, `ver_blocks`, `hor_blocks` are correctly populated.
	unsigned long error = compressBlock(dst, ver_blocks, hor_blocks, INT_MAX); // Inline: Call the compression function.
}

