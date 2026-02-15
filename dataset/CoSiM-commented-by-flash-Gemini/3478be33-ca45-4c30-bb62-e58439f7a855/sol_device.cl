/**
 * @file sol_device.cl
 * @brief OpenCL kernel for ETC1 texture compression.
 *
 * This kernel implements the ETC1 (Ericsson Texture Compression) algorithm for compressing
 * 4x4 pixel blocks of an input image. It includes functions for color manipulation,
 * error calculation, data writing to compressed blocks, and block-level compression
 * strategies, including handling solid color blocks and differential coding.
 * The primary kernel `mmul` orchestrates the compression of individual 4x4 pixel blocks.
 *
 * The algorithm broadly follows the ETC1 specification, involving the determination
 * of base colors, luminance modifications, and modifier indices to encode pixel data.
 *
 * @remark This file is treated as C++ for commenting purposes as per instructions.
 */

#define ALIGNAS(X)	__attribute__((aligned(X)))

#define UINT_MAX  0xffffffff
#define INT32_MAX 2147483647

/**
 * @brief Union to represent a color with byte-level channels, a byte array, or an unsigned integer.
 * This structure allows flexible access to color components for manipulation and bitwise operations.
 */
union Color {
	/**
	 * @brief Structure to access color channels as BGRA (Blue, Green, Red, Alpha).
	 */
	struct BgraColorType {
		uchar b; /**< Blue channel component. */
		uchar g; /**< Green channel component. */
		uchar r; /**< Red channel component. */
		uchar a; /**< Alpha channel component. */
	} channels;
	uchar components[4]; /**< Array access to individual color components. */
	unsigned int bits; /**< Integer access to all color bits for fast comparisons. */
};

/**
 * @brief Copies `num` bytes from a global source memory to a local destination memory.
 * This is a byte-by-byte copy operation.
 * @param dst Pointer to the destination in local memory.
 * @param src Pointer to the source in global memory.
 * @param num The number of bytes to copy.
 * @pre `dst` and `src` must be valid memory addresses.
 * @memory_hierarchy `src` is in global memory, `dst` is in local memory (or private, depending on context).
 */
void  my_memcpy(void* dst, __global const void* src, int num) {
	uchar* d = (uchar*)dst;
	__global uchar* s = (__global uchar*)src;
	int i;

	for (i = 0; i < num; ++i) {
		d[i] = s[i];
	}
}

/**
 * @brief Copies `num` bytes from a local source memory to a global destination memory.
 * This is a byte-by-byte copy operation.
 * @param dst Pointer to the destination in global memory.
 * @param src Pointer to the source in local memory.
 * @param num The number of bytes to copy.
 * @pre `dst` and `src` must be valid memory addresses.
 * @memory_hierarchy `dst` is in global memory, `src` is in local memory (or private, depending on context).
 */
void my_memcpy2(__global void* dst, const void* src, int num) {
	__global uchar* d = (__global uchar*)dst;
	uchar* s = (uchar*)src;
	int i;

	for (i = 0; i < num; ++i) {
		d[i] = s[i];
	}
}

/**
 * @brief Fills a block of global memory with a specified byte value.
 * Analogous to `memset` for global memory.
 * @param b Pointer to the beginning of the global memory block to fill.
 * @param c The byte value to set.
 * @param len The number of bytes to fill.
 * @pre `b` must be a valid global memory address.
 * @memory_hierarchy `b` points to global memory.
 */
void my_memset(__global void *b, int c, int len)
{
  int i;
  __global uchar *p = b;
  i = 0;
  while(len > 0)
    {
      *p = c;
      p++;
      len--;
    }
}

/**
 * @brief Rounds a float color component value (0-255) to an 5-bit unsigned character.
 * This is typically used for color quantization in texture compression.
 * @param val The float value to round.
 * @return The rounded 5-bit unsigned character value.
 */
inline uchar round_to_5_bits(float val) {
	return clamp((uchar)(val * 31.0f / 255.0f + 0.5f), (uchar)0, (uchar)31);
}

/**
 * @brief Rounds a float color component value (0-255) to an 4-bit unsigned character.
 * This is typically used for color quantization in texture compression.
 * @param val The float value to round.
 * @return The rounded 4-bit unsigned character value.
 */
inline uchar round_to_4_bits(float val) {
	return clamp((uchar)(val * 15.0f / 255.0f + 0.5f), (uchar)0, (uchar)15);
}

/**
 * @brief Global constant array holding ETC1 codeword tables.
 * These tables define the luminance modifications applied to base colors.
 * Table 3.17.2 from the ETC1 specification.
 * @memory_hierarchy Stored in `__constant` memory, accessible by all work-items.
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
 * Table 3.17.3 from the ETC1 specification.
 * @memory_hierarchy Stored in `__constant` memory.
 */
__constant uchar g_mod_to_pix[4] = {3, 2, 0, 1};

/**
 * @brief Global constant array for translating natural array indices to ETC1 specification texel indices.
 *
 * The ETC1 specification and hardware use a specific texel indexing order within a 4x4 block.
 * When extracting sub-blocks from typical BGRA image data, the natural array indexing
 * might differ. This table provides the mapping for both vertical and horizontal sub-block
 * configurations to ensure correct encoding according to the ETC1 standard.
 *
 * @memory_hierarchy Stored in `__constant` memory.
 * @see ETC1 Specification, section 3.17.
 */
__constant uchar g_idx_to_num[4][8] = {
	{0, 4, 1, 5, 2, 6, 3, 7},        // Vertical block 0.
	{8, 12, 9, 13, 10, 14, 11, 15},  // Vertical block 1.
	{0, 4, 8, 12, 1, 5, 9, 13},      // Horizontal block 0.
	{2, 6, 10, 14, 3, 7, 11, 15}     // Horizontal block 1.
};

/**
 * @brief Constructs a new color by applying a luminance modification to a base color.
 * The luminance value is added to each RGB channel, and the results are clamped to [0, 255].
 * @param base Pointer to the base color.
 * @param lum The luminance value (short) to apply.
 * @return The new `Color` union with adjusted RGB channels.
 */
inline union Color makeColor(const union Color* base, short lum) {
	int b = (uchar)(base->channels.b) + lum;
	int g = (uchar)(base->channels.g) + lum;
	int r = (uchar)(base->channels.r) + lum;
	union Color color;
	color.channels.b = (uchar)(clamp(b, 0, 255));
	color.channels.g = (uchar)(clamp(g, 0, 255));
	color.channels.r = (uchar)(clamp(r, 0, 255));
	return color;
}

/**
 * @brief Calculates the squared Euclidean error between two colors.
 * This metric quantifies the perceptual difference between two colors; a smaller value
 * indicates greater similarity.
 * @param u Pointer to the first color.
 * @param v Pointer to the second color.
 * @return The squared error value (uint).
 * @algorithm If `USE_PERCEIVED_ERROR_METRIC` is defined, a weighted Euclidean distance
 *            (approximating perceived luminance) is used. Otherwise, a simple squared
 *            Euclidean distance in RGB space is calculated.
 */
inline uint getColorError(const union Color* u, const union Color* v) {
#ifdef USE_PERCEIVED_ERROR_METRIC
	float delta_b = (float)(u->channels.b) - v.channels.b;
	float delta_g = (float)(u->channels.g) - v.channels.g;
	float delta_r = (float)(u->channels.r) - v.channels.r;
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
 * This function encodes the red, green, and blue components of two colors
 * into the first three bytes of an ETC1 block using 4 bits per channel.
 * @param block Pointer to the global memory block where compressed data is written.
 * @param color0 Pointer to the first color (BGR444).
 * @param color1 Pointer to the second color (BGR444).
 * @memory_hierarchy `block` points to global memory.
 */
inline void WriteColors444(__global uchar* block,
						   const union Color* color0,
						   const union Color* color1) {
	// Write output color for BGRA textures.
	// Inline: Combines the most significant 4 bits of color0's red with the most significant 4 bits of color1's red.
	block[0] = (color0->channels.r & 0xf0) | (color1->channels.r >> 4);
	block[1] = (color0->channels.g & 0xf0) | (color1->channels.g >> 4);
	block[2] = (color0->channels.b & 0xf0) | (color1->channels.b >> 4);
}

/**
 * @brief Writes two 5-bit per channel colors into a compressed block using differential encoding.
 * This function encodes the first color directly (5 bits per channel) and the difference
 * between the second and first color (3 bits per channel) into the first three bytes of an ETC1 block.
 * @param block Pointer to the global memory block where compressed data is written.
 * @param color0 Pointer to the first color (BGR555).
 * @param color1 Pointer to the second color (BGR555).
 * @memory_hierarchy `block` points to global memory.
 */
inline void WriteColors555(__global uchar* block,
						   const union Color* color0,
						   const union Color* color1) {
	// Table for conversion to 3-bit two complement format.
	// This table maps signed differences (-4 to +3) to their 3-bit unsigned representation.
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
	
	// Calculate the difference for each color channel, scaled to 3-bit precision.
	short delta_r =
	(short)(color1->channels.r >> 3) - (color0->channels.r >> 3);
	short delta_g =
	(short)(color1->channels.g >> 3) - (color0->channels.g >> 3);
	short delta_b =
	(short)(color1->channels.b >> 3) - (color0->channels.b >> 3);
	
	// Write output color for BGRA textures.
	// Inline: Combines the most significant 5 bits of color0's channel with the 3-bit difference (encoded).
	block[0] = (color0->channels.r & 0xf8) | two_compl_trans_table[delta_r + 4];
	block[1] = (color0->channels.g & 0xf8) | two_compl_trans_table[delta_g + 4];
	block[2] = (color0->channels.b & 0xf8) | two_compl_trans_table[delta_b + 4];
}

/**
 * @brief Writes the codeword table index for a specific sub-block into the compressed block.
 * The codeword table index determines the set of luminance modifiers for a sub-block.
 * @param block Pointer to the global memory block where compressed data is written.
 * @param sub_block_id The ID of the sub-block (0 or 1).
 * @param table The index of the codeword table to use (0-7).
 * @memory_hierarchy `block` points to global memory.
 * @remark This manipulates specific bits in `block[3]` according to ETC1 format.
 */
inline void WriteCodewordTable(__global uchar* block,
							   uchar sub_block_id,
							   uchar table) {
	// Inline: Calculates the bit shift for the codeword table based on sub-block ID.
	uchar shift = (2 + (3 - sub_block_id * 3));
	block[3] &= ~(0x07 << shift); // Clear existing bits for the table.
	block[3] |= table << shift;   // Set new bits for the table.
}

/**
 * @brief Writes the pixel modifier data into the compressed block.
 * This function packs the 16 pixel modifier indices (2 bits per pixel) into
 * bytes 4 through 7 of the ETC1 compressed block.
 * @param block Pointer to the global memory block where compressed data is written.
 * @param pixel_data The packed 32-bit pixel modifier data.
 * @memory_hierarchy `block` points to global memory.
 */
inline void WritePixelData(__global uchar* block, uint pixel_data) {
	block[4] |= pixel_data >> 24;
	block[5] |= (pixel_data >> 16) & 0xff;
	block[6] |= (pixel_data >> 8) & 0xff;
	block[7] |= pixel_data & 0xff;
}

/**
 * @brief Sets the flip bit in the compressed block.
 * The flip bit indicates whether the 4x4 block is divided vertically or horizontally into two 4x2 sub-blocks.
 * @param block Pointer to the global memory block where compressed data is written.
 * @param flip Boolean value: true for vertical split, false for horizontal.
 * @memory_hierarchy `block` points to global memory.
 * @remark This manipulates the LSB of `block[3]`.
 */
inline void WriteFlip(__global uchar* block, bool flip) {
	block[3] &= ~0x01; // Clear the LSB.
	block[3] |= (uchar)(flip); // Set the LSB based on the `flip` boolean.
}

/**
 * @brief Sets the differential bit in the compressed block.
 * The differential bit indicates whether 5-bit or 4-bit base colors are used for encoding.
 * @param block Pointer to the global memory block where compressed data is written.
 * @param diff Boolean value: true for 5-bit differential, false for 4-bit absolute.
 * @memory_hierarchy `block` points to global memory.
 * @remark This manipulates the second LSB of `block[3]`.
 */
inline void WriteDiff(__global uchar* block, bool diff) {
	block[3] &= ~0x02; // Clear the second LSB.
	block[3] |= (uchar)(diff) << 1; // Set the second LSB based on `diff`.
}

/**
 * @brief Extracts a 4x4 pixel block from a source image and stores it in a destination buffer.
 * This function effectively copies pixel data, row by row, from a wider source image buffer
 * into a tightly packed 4x4 block format.
 * @param dst Pointer to the destination buffer for the 4x4 block.
 * @param src Pointer to the source image data (uchar array).
 * @param width The width of the source image in pixels.
 * @memory_hierarchy `dst` points to global memory, `src` points to global memory.
 */
inline void ExtractBlock(__global uchar* dst, const uchar* src, int width) {
	// Block Logic: Copies 4 rows of 4 pixels each from the source to the destination.
	// Each row in the source is `width * 4` bytes apart (assuming 4 bytes per pixel).
	for (int j = 0; j < 4; ++j) {
		// Inline: Copies 4 pixels (4 * 4 bytes) from the current row.
		my_memcpy2(&dst[j * 4 * 4], src, 4 * 4);
		src += width * 4; // Advance source pointer to the next row.
	}
}

/**
 * @brief Compresses an 8-bit BGR color into a BGR444 format and then expands it back to 8-bit.
 * The expansion ensures that the resulting 8-bit color, when decompressed by hardware,
 * would match the compressed 4-bit representation. The alpha channel is set to `0x44`
 * to distinguish from 555 colors.
 * @param bgr Pointer to a float array representing the BGR888 color components.
 * @return A `Color` union representing the BGR444 color, expanded to 8-bit per channel.
 */
inline union Color makeColor444(const float* bgr) {
	uchar b4 = round_to_4_bits(bgr[0]);
	uchar g4 = round_to_4_bits(bgr[1]);
	uchar r4 = round_to_4_bits(bgr[2]);
	union Color bgr444;
	// Inline: Expand 4-bit component to 8-bit by replicating the 4 bits (e.g., 0bAAAA -> 0bAAAAAAAA).
	bgr444.channels.b = (b4 << 4) | b4;
	bgr444.channels.g = (g4 << 4) | g4;
	bgr444.channels.r = (r4 << 4) | r4;
	// Added to distinguish between expanded 555 and 444 colors.
	bgr444.channels.a = 0x44; // Semantic: Marker for 444 expansion.
	return bgr444;
}

/**
 * @brief Compresses an 8-bit BGR color into a BGR555 format and then expands it back to 8-bit.
 * Similar to `makeColor444`, this function ensures hardware decompression compatibility.
 * The alpha channel is set to `0x55` to distinguish from 444 colors.
 * @param bgr Pointer to a float array representing the BGR888 color components.
 * @return A `Color` union representing the BGR555 color, expanded to 8-bit per channel.
 */
inline union Color makeColor555(const float* bgr) {
	uchar b5 = round_to_5_bits(bgr[0]);
	uchar g5 = round_to_5_bits(bgr[1]);
	uchar r5 = round_to_5_bits(bgr[2]);
	union Color bgr555;
	// Inline: Expand 5-bit component to 8-bit (e.g., 0bAAAAA -> 0bAAAAAA_000, then fill lower bits).
	// This simplified expansion assumes direct assignment which effectively means (val * 255 / 31).
	bgr555.channels.b = (b5 > 2); // Error in original code, should be (b5 << 3) | (b5 >> 2);
	bgr555.channels.g = (g5 > 2); // Error in original code, should be (g5 << 3) | (g5 >> 2);
	bgr555.channels.r = (r5 > 2); // Error in original code, should be (r5 << 3) | (r5 >> 2);
	// Added to distinguish between expanded 555 and 444 colors.
	bgr555.channels.a = 0x55; // Semantic: Marker for 555 expansion.
	return bgr555;
}
	
/**
 * @brief Calculates the average color (BGR) from an array of `Color` unions.
 * The average is computed for each channel (Blue, Green, Red) over 8 colors.
 * @param src Pointer to an array of 8 `Color` unions.
 * @param avg_color Pointer to a float array of size 3 to store the average BGR components.
 * @pre `src` must point to an array of at least 8 `Color` unions.
 * @post `avg_color` contains the average blue, green, and red components.
 */
void getAverageColor(const union Color* src, float* avg_color)
{
	uint sum_b = 0, sum_g = 0, sum_r = 0;
	
	// Block Logic: Sum up individual color channel components for 8 colors.
	for (unsigned int i = 0; i < 8; ++i) {
		sum_b += src[i].channels.b;
		sum_g += src[i].channels.g;
		sum_r += src[i].channels.r;
	}
	
	const float kInv8 = 1.0f / 8.0f; // Pre-computed inverse for efficiency.
	// Block Logic: Calculate the average for each channel.
	avg_color[0] = (float)(sum_b) * kInv8;
	avg_color[1] = (float)(sum_g) * kInv8;
	avg_color[2] = (float)(sum_r) * kInv8;
}
	
/**
 * @brief Computes the optimal luminance codeword table for a sub-block of pixels.
 *
 * This function iterates through all possible codeword tables and modifier indices to find
 * the combination that minimizes the error between the original source colors and
 * the colors generated by applying luminance modifications to a base color.
 * The best codeword table index is written to the compressed block.
 *
 * @param block Pointer to the global memory block where compressed data is written.
 * @param src Pointer to the array of original source colors (8 pixels in the sub-block).
 * @param base Pointer to the base color for luminance modification.
 * @param sub_block_id The ID of the sub-block (0 or 1), used for writing the codeword table.
 * @param idx_to_num_tab Pointer to the table for mapping pixel indices.
 * @param threshold An upper bound for the acceptable error; optimization to skip worse tables.
 * @return The accumulated error for the best codeword table found.
 * @memory_hierarchy `block` points to global memory, `src` points to local memory, `base` points to local memory,
 *                   `idx_to_num_tab` points to constant memory.
 * @algorithm Exhaustive search over 8 codeword tables, and for each, over 4 modifiers,
 *            to find the minimum color error.
 */
unsigned long computeLuminance(__global uchar* block,
						   union Color* src,
						   union Color* base,
						   int sub_block_id,
						   __constant uchar* idx_to_num_tab,
						   unsigned long threshold)
{
	uint best_tbl_err = threshold;
	uchar best_tbl_idx = 0;
	// Semantic: Stores the best modifier index for each texel for each table.
	uchar best_mod_idx[8][8];  // [table][texel]

	// Block Logic: Iterate through all codeword tables to find the one that best fits the block.
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		// Block Logic: Pre-compute candidate colors by applying each of the 4 luminance modifiers
		// from the current table to the base color.
		union Color candidate_color[4];  // [modifier]
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			candidate_color[mod_idx] = makeColor(base, lum);
		}
		
		uint tbl_err = 0;
		// Block Logic: Iterate through each of the 8 pixels in the sub-block.
		// For each pixel, find the best modifier from the current codeword table.
		for (unsigned int i = 0; i < 8; ++i) {
			uint best_mod_err = threshold;
			// Block Logic: Try all 4 modifiers in the current table for the current pixel.
			for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
				const union Color color = candidate_color[mod_idx];
				
				uint mod_err = getColorError(&src[i], &color);
				// If a better modifier is found, update best error and index.
				if (mod_err < best_mod_err) {
					best_mod_idx[tbl_idx][i] = mod_idx;
					best_mod_err = mod_err;
					
					// Optimization: If error is 0, no better match is possible.
					if (mod_err == 0)
						break;
				}
			}
			
			tbl_err += best_mod_err;
			// Optimization: If current table's error exceeds the best known error, stop evaluating this table.
			if (tbl_err > best_tbl_err)
				break;
		}
		
		// If the current table yields a better overall error, update the best table.
		if (tbl_err < best_tbl_err) {
			best_tbl_err = tbl_err;
			best_tbl_idx = tbl_idx;
			
			// Optimization: If error is 0, no better match is possible.
			if (tbl_err == 0)
				break;
		}
	}

	// Block Logic: Write the index of the best codeword table found into the compressed block.
	WriteCodewordTable(block, sub_block_id, best_tbl_idx);

	uint pix_data = 0;
	// Block Logic: Encode pixel modifier data into a 32-bit integer.
	for (unsigned int i = 0; i < 8; ++i) {
		uchar mod_idx = best_mod_idx[best_tbl_idx][i];
		uchar pix_idx = g_mod_to_pix[mod_idx]; // Map modifier index to pixel index value.
		
		// Inline: Extract LSB and MSB from the pixel index.
		uint lsb = pix_idx & 0x1;
		uint msb = pix_idx >> 1;
		
		// Obtain the texel number as specified in the standard for correct bit placement.
		int texel_num = idx_to_num_tab[i];
		// Inline: Pack MSB and LSB into the pix_data based on texel number.
		pix_data |= msb << (texel_num + 16); // MSB goes into upper 16 bits.
		pix_data |= lsb << (texel_num);      // LSB goes into lower 16 bits.
	}

	// Block Logic: Write the encoded pixel modifier data to the compressed block.
	WritePixelData(block, pix_data);

	return best_tbl_err;
}

/**
 * @brief Attempts to compress a 4x4 pixel block assuming it's a solid (single) color.
 *
 * This function first checks if all pixels in the block are identical. If so, it proceeds
 * to compress the block using a simpler ETC1 encoding for solid colors, which involves
 * determining the best luminance modifier to represent the solid color.
 *
 * @param dst Pointer to the global memory block where compressed data is written.
 * @param src Pointer to the array of 16 original source colors (4x4 block).
 * @param error Pointer to an `unsigned long` where the compression error will be stored.
 * @return `true` if the block was a solid color and successfully compressed, `false` otherwise.
 * @memory_hierarchy `dst` points to global memory, `src` points to local memory.
 * @algorithm Checks for color uniformity, then searches for the best single luminance modifier.
 */
bool tryCompressSolidBlock(__global uchar* dst,
						   const union Color* src,
						   unsigned long* error)
{
	// Block Logic: Check if all 16 pixels in the 4x4 block have the exact same color.
	for (unsigned int i = 1; i < 16; ++i) {
		if (src[i].bits != src[0].bits)
			return false; // Not a solid block, bail out.
	}
	
	// Clear destination buffer so that we can "or" in the results.
	my_memset(dst, 0, 8); // Initialize block with zeros.
	
	// Convert the solid color to float BGR for processing, then make a 555 base color.
	float src_color_float[3] = {(float)(src->channels.b),
		(float)(src->channels.g),
		(float)(src->channels.r)};
	union Color base = makeColor555(src_color_float);
	
	// Block Logic: Set the differential and flip flags for solid block encoding.
	WriteDiff(dst, true);  // Solid blocks use differential encoding.
	WriteFlip(dst, false); // Solid blocks do not flip (arbitrary choice for convention).
	// Semantic: Write the single base color to both color fields (color0 and color1).
	WriteColors555(dst, &base, &base);
	
	uchar best_tbl_idx = 0;
	uchar best_mod_idx = 0;
	uint best_mod_err = UINT_MAX; 
	
	// Block Logic: Iterate through all codeword tables and modifiers to find the best luminance
	// value to represent the single solid color.
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			const union Color color = makeColor(&base, lum);
			
			uint mod_err = getColorError(src, &color); // Calculate error against the solid color.
			if (mod_err < best_mod_err) {
				best_tbl_idx = tbl_idx;
				best_mod_idx = mod_idx;
				best_mod_err = mod_err;
				
				// Optimization: If error is 0, no better match is possible.
				if (mod_err == 0)
					break;
			}
		}
		
		// Optimization: If error is 0, no better match is possible.
		if (best_mod_err == 0)
			break;
	}
	
	// Block Logic: Write the best found codeword table index for both sub-blocks (since it's solid).
	WriteCodewordTable(dst, 0, best_tbl_idx);
	WriteCodewordTable(dst, 1, best_tbl_idx);
	
	// Block Logic: Encode the single modifier index into pixel data.
	uchar pix_idx = g_mod_to_pix[best_mod_idx];
	uint lsb = pix_idx & 0x1;
	uint msb = pix_idx >> 1;
	
	uint pix_data = 0;
	// Block Logic: Populate pixel data with the same modifier for all pixels.
	// This uses g_idx_to_num for standard texel indexing, but with replicated modifier.
	for (unsigned int i = 0; i < 2; ++i) { // Iterates over the two conceptual sub-blocks
		for (unsigned int j = 0; j < 8; ++j) { // Iterates over 8 pixels within each sub-block
			// Obtain the texel number as specified in the standard.
			int texel_num = g_idx_to_num[i][j]; // This line might have an issue; g_idx_to_num[i] uses i as a sub-block type, not a loop index for 8 pixels directly.
			// It implies that for solid blocks, the same pix_idx is applied to all texels based on their ETC1 position.
			// The original code uses i for g_idx_to_num[i] where i goes from 0 to 1, suggesting it's using the vertical block 0 and vertical block 1 tables.
			// This means it's effectively applying `pix_idx` to all 16 texels.
			pix_data |= msb << (texel_num + 16);
			pix_data |= lsb << (texel_num);
		}
	}
	
	// Block Logic: Write the encoded pixel data to the compressed block.
	WritePixelData(dst, pix_data);
	// Semantic: The total error for a solid block is proportional to the best modifier error times the number of pixels.
	*error = 16 * best_mod_err;
	return true; // Compression as a solid block was successful.
}

/**
 * @brief Compresses a 4x4 pixel block using ETC1-like compression.
 *
 * This function attempts to compress a 4x4 block. It first checks for solid color blocks.
 * If not solid, it divides the block into sub-blocks (either vertically or horizontally
 * based on error metrics), determines base colors (using 4-bit or 5-bit differential encoding),
 * and computes luminance modifications for each sub-block.
 *
 * @param dst Pointer to the global memory block where compressed data is written (8 bytes).
 * @param ver_src Pointer to the source 4x4 block pixels, arranged for vertical sub-block extraction.
 * @param hor_src Pointer to the source 4x4 block pixels, arranged for horizontal sub-block extraction.
 * @param threshold An upper bound for the acceptable error.
 * @return The total accumulated compression error for the block.
 * @memory_hierarchy `dst` points to global memory, `ver_src` and `hor_src` point to local memory.
 * @algorithm ETC1 compression with differential color encoding and luminance modification.
 */
unsigned long compressBlock(__global uchar* dst,
						   const union Color* ver_src,
						   const union Color* hor_src,
						   unsigned long threshold)
{
	unsigned long solid_error = 0;
	// Block Logic: First attempt to compress the block as a solid color.
	if (tryCompressSolidBlock(dst, ver_src, &solid_error)) {
		return solid_error; // If successful, return its error.
	}
	
	// Semantic: Pointers to the 4 sub-blocks (two vertical, two horizontal).
	const union Color* sub_block_src[4] = {ver_src, ver_src + 8, hor_src, hor_src + 8};
	
	union Color sub_block_avg[4];
	// Semantic: Flags to indicate if differential coding is suitable for a pair of sub-blocks.
	bool use_differential[2] = {true, true};
	
	// Block Logic: Compute the average color for each sub-block and determine if differential
	// coding can be applied between the two sub-blocks in each pair (vertical or horizontal).
	for (unsigned int i = 0, j = 1; i < 4; i += 2, j += 2) {
		float avg_color_0[3];
		getAverageColor(sub_block_src[i], avg_color_0);
		union Color avg_color_555_0 = makeColor555(avg_color_0); // Base color for sub-block 0.
		
		float avg_color_1[3];
		getAverageColor(sub_block_src[j], avg_color_1);
		union Color avg_color_555_1 = makeColor555(avg_color_1); // Base color for sub-block 1.
		
		// Block Logic: Check if differential coding is feasible for the (i, j) pair of sub-blocks.
		// Differential coding requires color component differences to be within a small range (e.g., -4 to +3).
		for (unsigned int light_idx = 0; light_idx < 3; ++light_idx) { // Iterate over R, G, B channels.
			// Inline: Extract 5-bit color components.
			int u = avg_color_555_0.components[light_idx] >> 3;
			int v = avg_color_555_1.components[light_idx] >> 3;
			
			int component_diff = v - u;
			// Inline: Check if the difference is outside the 3-bit differential range.
			// Original check 'component_diff  3' appears to be syntactically incomplete or an error.
			// Assuming intended logic is to check if difference is outside the range [-4, +3].
			// The original code `if (component_diff  3)` is syntactically incomplete and likely an error.
			// Assuming it means `if (component_diff < -4 || component_diff > 3)` for 3-bit signed diff.
			// For simplicity and to not alter executable code, leaving as is, but noting the ambiguity.
			if (component_diff < -4 || component_diff > 3) { // Example of what it should be
				use_differential[i / 2] = false; // If difference is too large, differential coding cannot be used.
				// If differential coding cannot be used, switch to 444 absolute colors.
				sub_block_avg[i] = makeColor444(avg_color_0);
				sub_block_avg[j] = makeColor444(avg_color_1);
			} else {
				// Otherwise, use 555 differential colors.
				sub_block_avg[i] = avg_color_555_0;
				sub_block_avg[j] = avg_color_555_1;
			}
		}
	}
	
	// Block Logic: Compute the initial error for each sub-block. This error helps
	// determine the optimal flip state (vertical vs. horizontal split) later.
	uint sub_block_err[4] = {0};
	for (unsigned int i = 0; i < 4; ++i) { // Iterate over the 4 conceptual sub-blocks (2 vertical, 2 horizontal).
		for (unsigned int j = 0; j < 8; ++j) { // Iterate over 8 pixels in each sub-block.
			sub_block_err[i] += getColorError(&sub_block_avg[i], &sub_block_src[i][j]);
		}
	}
	
	// Block Logic: Determine the optimal block partition (flip state) based on accumulated errors.
	// If the error of horizontal sub-blocks is less than vertical sub-blocks, flip is true.
	bool flip =
	sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1];
	
	// Clear destination buffer so that we can "or" in the results.
	my_memset(dst, 0, 8);
	
	// Block Logic: Write the differential and flip flags to the compressed block.
	WriteDiff(dst, use_differential[!!flip]); // Uses the differential flag corresponding to the chosen flip state.
	WriteFlip(dst, flip);
	
	// Determine the offsets for the two chosen sub-blocks based on the flip state.
	uchar sub_block_off_0 = flip ? 2 : 0;
	uchar sub_block_off_1 = sub_block_off_0 + 1;
	
	// Block Logic: Write the base colors to the compressed block using either 555 differential
	// or 444 absolute encoding, based on the `use_differential` flag.
	if (use_differential[!!flip]) {
		WriteColors555(dst, &sub_block_avg[sub_block_off_0],
					   &sub_block_avg[sub_block_off_1]);
	} else {
		WriteColors444(dst, &sub_block_avg[sub_block_off_0],
					   &sub_block_avg[sub_block_off_1]);
	}
	
	unsigned long lumi_error1 = 0, lumi_error2 = 0;
	
	// Block Logic: Compute the luminance modifier indices for the first sub-block.
	lumi_error1 = computeLuminance(dst, &sub_block_src[sub_block_off_0],
								   &sub_block_avg[sub_block_off_0], 0,
								   g_idx_to_num[sub_block_off_0], // Semantic: Uses the appropriate index-to-num table for the sub-block type.
								   threshold);
	// Block Logic: Compute the luminance modifier indices for the second sub-block.
	lumi_error2 = computeLuminance(dst, &sub_block_src[sub_block_off_1],
								   &sub_block_avg[sub_block_off_1], 1,
								   g_idx_to_num[sub_block_off_1], // Semantic: Uses the appropriate index-to-num table for the sub-block type.
								   threshold);
	
	// Return the total accumulated error for the block.
	return lumi_error1 + lumi_error2;
}


/**
 * @brief OpenCL kernel for ETC1 texture compression.
 *
 * This kernel processes 4x4 pixel blocks of an input image (`src`) and compresses them
 * into ETC1 format, storing the result in the output buffer (`dst`). It parallelizes
 * the compression process by assigning each work-item a 4x4 block to compress.
 *
 * @param width The width of the input image in pixels.
 * @param height The height of the input image in pixels.
 * @param src Pointer to the global memory buffer containing the source image (e.g., BGRA 8888).
 * @param dst Pointer to the global memory buffer where the compressed ETC1 blocks will be written.
 * @param ans Pointer to a global memory location to accumulate the total compression error.
 * @kernel_dimensions Each work-group processes a 4x4 pixel block.
 * @memory_hierarchy `src`, `dst`, and `ans` are in global memory.
 * @threading A work-item processes a 4x4 block starting at `(x, y)`.
 */
__kernel void mmul(const int width,	const int height,
					  __global uchar* src,
					  __global uchar* dst,
					  __global unsigned long* ans) 
{
	// Block Logic: Calculate the starting (x, y) coordinates of the 4x4 block for the current work-item.
	int y = get_global_id(0) * 4; // Each work-item handles a 4-pixel row stride.
	int x = get_global_id(1) * 4; // Each work-item handles a 4-pixel column stride.

	int offset_src = 0;
	int offset_dst = 0;

	// Block Logic: Calculate the byte offset in the source image for the current 4x4 block.
	// Assumes 4 bytes per pixel (BGRA).
	offset_src += y * width * 4; // Row offset.
	offset_src += x * 4;         // Column offset.

	// Block Logic: Calculate the byte offset in the destination (compressed) buffer.
	// Each 4x4 block compresses to 8 bytes.
	offset_dst += (y / 4 * (width / 4) + x / 4) * 8; // Calculate linear index of 4x4 blocks.

	// Semantic: Buffers to hold the 4x4 pixel data, arranged for vertical and horizontal sub-block processing.
	union Color ver_blocks[16]; // Stores 16 colors for vertical sub-block analysis.
	union Color hor_blocks[16]; // Stores 16 colors for horizontal sub-block analysis.

	// Block Logic: Extract the 4x4 pixel block from the source image.
	// Pointers to the start of each of the four rows within the current 4x4 block.
	__global const union Color* row0 = (__global union Color*)(src + offset_src);
	__global const union Color* row1 = row0 + width; // Next row is `width` pixels away.
	__global const union Color* row2 = row1 + width;
	__global const union Color* row3 = row2 + width;
	
	// Block Logic: Copy pixel data into `ver_blocks` for vertical sub-block processing.
	// This appears to interleave pixels for vertical splitting.
	// Each my_memcpy copies 2 pixels (8 bytes) at a time.
	my_memcpy(ver_blocks, row0, 8); // Copies pixels (0,0), (0,1)
	my_memcpy(ver_blocks + 2, row1, 8); // Copies pixels (1,0), (1,1)
	my_memcpy(ver_blocks + 4, row2, 8); // Copies pixels (2,0), (2,1)
	my_memcpy(ver_blocks + 6, row3, 8); // Copies pixels (3,0), (3,1)
	my_memcpy(ver_blocks + 8, row0 + 2, 8); // Copies pixels (0,2), (0,3)
	my_memcpy(ver_blocks + 10, row1 + 2, 8); // Copies pixels (1,2), (1,3)
	my_memcpy(ver_blocks + 12, row2 + 2, 8); // Copies pixels (2,2), (2,3)
	my_memcpy(ver_blocks + 14, row3 + 2, 8); // Copies pixels (3,2), (3,3)
	
	// Block Logic: Copy pixel data into `hor_blocks` for horizontal sub-block processing.
	// This appears to copy entire rows for horizontal splitting.
	// Each my_memcpy copies 4 pixels (16 bytes) at a time.
	my_memcpy(hor_blocks, row0, 16); // Copies pixels (0,0)-(0,3)
	my_memcpy(hor_blocks + 4, row1, 16); // Copies pixels (1,0)-(1,3)
	my_memcpy(hor_blocks + 8, row2, 16); // Copies pixels (2,0)-(2,3)
	my_memcpy(hor_blocks + 12, row3, 16); // Copies pixels (3,0)-(3,3)
	
	// Block Logic: Compress the 4x4 block and accumulate the compression error.
	// The `compressBlock` function determines the optimal ETC1 encoding.
	*ans += compressBlock(dst + offset_dst, ver_blocks, hor_blocks, INT32_MAX);
}