


/**



 * @file compress.cl



 * @brief This file contains OpenCL kernel code for compressing image blocks,



 *        along with C++ host code for managing the OpenCL context and executing



 *        the compression kernel. It implements a texture compression algorithm,



 *        likely a variant of BCn/DXT formats, by quantizing colors and encoding



 *        luminance and pixel data.



 */











/**



 * @union Color



 * @brief Represents a pixel color using a union for flexible access.



 *



 * This union allows accessing color components as individual BGRA channels,



 * as an array of unsigned characters, or as a single 32-bit unsigned integer.



 */



typedef union Tag {



	struct BgraColorType {



		unsigned char b; /**< Blue channel component (8-bit). */



		unsigned char g; /**< Green channel component (8-bit). */



		unsigned char r; /**< Red channel component (8-bit). */



		unsigned char a; /**< Alpha channel component (8-bit). */



	} channels; /**< Access color components by name. */



	unsigned char components[4]; /**< Access color components as an array. */



	unsigned int bits; /**< Access the entire color as a 32-bit integer. */



} Color;







/**



 * @brief Clamps an integer value within a specified minimum and maximum range.



 * @param val The input integer value.



 * @param min The minimum allowed value.



 * @param max The maximum allowed value.



 * @return The clamped integer value.



 * @note There appears to be a typo in the implementation: `val max ? max : val)`



 *       should likely be `(val > max ? max : val)`. Assuming `clamp` is defined elsewhere



 *       or this is a custom macro/function.



 */



int clamp1(int val, int min, int max) {



	return val  max ? max : val);



}











/**



 * @brief Clamps an unsigned integer value within a specified minimum and maximum range.



 * @param val The input unsigned integer value.



 * @param min The minimum allowed value.



 * @param max The maximum allowed value.



 * @return The clamped unsigned integer value.



 * @note There appears to be a typo in the implementation: `val max ? max : val)`



 *       should likely be `(val > max ? max : val)`. Assuming `clamp` is defined elsewhere



 *       or this is a custom macro/function.



 */



unsigned int clamp2(unsigned int val, unsigned int min, unsigned int max) {



	return val  max ? max : val);



}







/**



 * @brief Rounds a floating-point color component (0-255 range) to a 5-bit unsigned character.



 *



 * This function scales the input float value to a 0-31 range (5 bits) and rounds it.



 * It's commonly used in color compression to reduce color precision.



 *



 * @param val The floating-point color component value (expected to be 0-255).



 * @return The rounded 5-bit unsigned character value.



 */



unsigned char round_to_5_bits(float val) {



	return (unsigned char) clamp2(val * 31.0f / 255.0f + 0.5f, 0, 31);



}







/**



 * @brief Rounds a floating-point color component (0-255 range) to a 4-bit unsigned character.



 *



 * This function scales the input float value to a 0-15 range (4 bits) and rounds it.



 * It's commonly used in color compression to reduce color precision.



 *



 * @param val The floating-point color component value (expected to be 0-255).



 * @return The rounded 4-bit unsigned character value.



 */



unsigned char round_to_4_bits(float val) {



	return (unsigned char) clamp2(val * 15.0f / 255.0f + 0.5f, 0, 15);



}







/**



 * @brief Global constant table containing pre-defined luminance codewords.



 *



 * These codewords are short integer offsets applied to base colors to generate



 * a small color palette (e.g., 4 colors) for a compressed block. The 8 rows



 * likely represent different "tables" or "sets" of luminance modifiers,



 * and the 4 columns correspond to different modulation indices.



 */



__constant short g_codeword_tables[8][4] = {



	{-8, -2, 2, 8},    /**< Codeword table 0 */



	{-17, -5, 5, 17},  /**< Codeword table 1 */



	{-29, -9, 9, 29},  /**< Codeword table 2 */



	{-42, -13, 13, 42},/**< Codeword table 3 */



	{-60, -18, 18, 60},/**< Codeword table 4 */











	{-80, -24, 24, 80},  /**< Codeword table 5 */



	{-106, -33, 33, 106},/**< Codeword table 6 */



	{-183, -47, 47, 183} /**< Codeword table 7 */



};







/**



 * @brief Global constant array mapping modulation indices to pixel indices.



 *



 * This table reorders the modulation indices (derived from `g_codeword_tables`)



 * to a specific output pixel index order, potentially for optimization or



 * compatibility with a particular compression format.



 */



__constant unsigned char g_mod_to_pix[4] = {3, 2, 0, 1};







/**



 * @brief Global constant lookup table for remapping pixel indices within a block.



 *



 * This 2D array provides a mapping from a linear pixel index (0-7) within a



 * sub-block to a different numerical index, possibly for arranging pixels



 * into a specific bit pattern for storage in the compressed block.



 */



__constant unsigned char g_idx_to_num[4][8] = {



	{0, 4, 1, 5, 2, 6, 3, 7},        /**< Index mapping for sub-block 0 */



	{8, 12, 9, 13, 10, 14, 11, 15},  /**< Index mapping for sub-block 1 */











	{0, 4, 8, 12, 1, 5, 9, 13},      /**< Index mapping for sub-block 2 */



	{2, 6, 10, 14, 3, 7, 11, 15}     /**< Index mapping for sub-block 3 */



};







/**



 * @brief Creates a new Color by applying a luminance offset to a base color.



 *



 * This function takes a base color and adds a `lum` value (luminance offset)



 * to its R, G, and B channels. The resulting channel values are clamped



 * to the valid 0-255 range.



 *



 * @param base The initial Color structure.



 * @param lum The luminance offset to apply (can be positive or negative).



 * @return A new Color structure with adjusted luminance.



 * @note Assumes `clamp` function is available in the global scope.



 */



Color makeColor(const Color base, short lum) {



	int b = (int) base.channels.b + lum;



	int g = (int) base.channels.g + lum;



	int r = (int) base.channels.r + lum;



	Color color;



	color.channels.b = (unsigned char) clamp(b, 0, 255);



	color.channels.g = (unsigned char) clamp(g, 0, 255);



	color.channels.r = (unsigned char) clamp(r, 0, 255);



	return color;



}







#define USE_PERCEIVED_ERROR_METRIC /**< Macro to enable a perceptually weighted error metric. */











/**



 * @brief Calculates the squared color difference (error) between two colors.



 *



 * This function can use either a perceptually weighted error metric (similar to YCbCr



 * weighting) if `USE_PERCEIVED_ERROR_METRIC` is defined, or a simple squared



 * Euclidean distance in RGB color space. The perceptually weighted metric



 * prioritizes errors in components human vision is more sensitive to.



 *



 * @param u The first Color.



 * @param v The second Color.



 * @return The calculated squared color error as an unsigned integer.



 */



unsigned int getColorError(const Color u, const Color v) {



#ifdef USE_PERCEIVED_ERROR_METRIC



	float delta_b = (float) u.channels.b - v.channels.b;



	float delta_g = (float) u.channels.g - v.channels.g;



	float delta_r = (float) u.channels.r - v.channels.r;



	// Perceptually weighted squared error sum (approx. YCbCr weighting).



	return (unsigned int) (0.299f * delta_b * delta_b +



								 0.587f * delta_g * delta_g +



								 0.114f * delta_r * delta_r);



#else



	// Simple squared Euclidean distance in RGB color space.



	int delta_b = static_cast<int>(u.channels.b) - v.channels.b;



	int delta_g = static_cast<int>(u.channels.g) - v.channels.g;



	int delta_r = static_cast<int>(u.channels.r) - v.channels.r;











	return delta_b * delta_b + delta_g * delta_g + delta_r * delta_r;



#endif



}







/**



 * @brief Writes two 4-bit per channel colors into a destination byte block.



 *



 * This function encodes two colors (`color0` and `color1`), where each RGB



 * channel is represented by 4 bits. It packs these 4-bit components into



 * a compressed byte format. This is typical for certain texture compression



 * formats that use a reduced color precision.



 *



 * @param block A pointer to the global memory block where the compressed colors will be written.



 * @param color0 The first Color to encode.



 * @param color1 The second Color to encode.



 */



void WriteColors444(global unsigned char* block,



									 const Color color0,



									 const Color color1) {



	// Combines the higher 4 bits of color0.r with the higher 4 bits of color1.r (shifted).



	block[0] = (color0.channels.r & 0xf0) | (color1.channels.r >> 4);



	// Combines the higher 4 bits of color0.g with the higher 4 bits of color1.g (shifted).



	block[1] = (color0.channels.g & 0xf0) | (color1.channels.g >> 4);











	// Combines the higher 4 bits of color0.b with the higher 4 bits of color1.b (shifted).



	block[2] = (color0.channels.b & 0xf0) | (color1.channels.b >> 4);



}







/**



 * @brief Writes two 5-bit per channel colors (or their differential encoding)



 *        into a destination byte block.



 *



 * This function encodes `color0` (5 bits per channel) and the *difference*



 * between `color1` and `color0` (also 5-bit precision). The differences are



 * encoded using a 2's complement representation, suitable for differential



 * encoding in texture compression schemes (e.g., BC4/BC5).



 *



 * @param block A pointer to the global memory block where the compressed colors will be written.



 * @param color0 The base Color (5-bit precision).



 * @param color1 The differential Color (5-bit precision).



 */



void WriteColors555(global unsigned char* block,



						   const Color color0,



						   const Color color1) {



	// Lookup table for 2's complement conversion of 3-bit differences to 5-bit representation.



	// This maps a signed difference in [-4, 3] to an unsigned value [0, 7] for storage.



	const unsigned char two_compl_trans_table[8] = {



		4,  /**< Represents -4 */



		5,  /**< Represents -3 */



		6,  /**< Represents -2 */



		7,  /**< Represents -1 */



		0,  /**< Represents  0 */



		1,  /**< Represents  1 */



		2,  /**< Represents  2 */



		3,  /**< Represents  3 */



	};



	



	// Calculate 5-bit differences for R, G, B channels. (RHS>>3 effectively gets 5-bit values assuming 0-255 input)



	short delta_r =



	(short) (color1.channels.r >> 3) - (color0.channels.r >> 3);



	short delta_g =



	(short) (color1.channels.g >> 3) - (color0.channels.g >> 3);



	short delta_b =



	(short) (color1.channels.b >> 3) - (color0.channels.b >> 3);



	



	// Packs the 5 most significant bits of color0.r and the 3-bit encoded delta_r.



	// The delta is biased by +4 to index into `two_compl_trans_table`.



	block[0] = (color0.channels.r & 0xf8) | two_compl_trans_table[delta_r + 4];











	block[1] = (color0.channels.g & 0xf8) | two_compl_trans_table[delta_g + 4];



	block[2] = (color0.channels.b & 0xf8) | two_compl_trans_table[delta_b + 4];



}







/**



 * @brief Writes the selected codeword table index into the output block.



 *



 * This function updates specific bits within the 3rd byte of the block



 * to store the chosen codeword table (`table`) for a given sub-block.



 * This table determines the luminance offsets used for reconstruction.



 *



 * @param block A pointer to the global memory block.



 * @param sub_block_id The ID of the sub-block (0 or 1).



 * @param table The index of the codeword table to write (0-7).



 */



void WriteCodewordTable(global unsigned char* block,



							   unsigned char sub_block_id,



							   unsigned char table) {



	// Calculates the shift amount for inserting the table index into `block[3]`.



	// The shift depends on the `sub_block_id` to target different bit fields.



	unsigned char shift = (2 + (3 - sub_block_id * 3)); // For sub_block_id 0, shift is 8. For sub_block_id 1, shift is 5.



	



	// Clears the target bits in `block[3]` before inserting the new table value.



	block[3] &= ~(0x07 << shift); // 0x07 is 111b, clearing 3 bits.



	// Sets the table index into the cleared bits.



	block[3] |= table << shift;



}







/**



 * @brief Writes 32-bit pixel modulation data into the output block.



 *



 * This function takes a 32-bit unsigned integer representing pixel modulation



 * data (typically 2 bits per pixel, indicating which color from the palette



 * to use) and writes it into bytes 4, 5, 6, and 7 of the compressed block.



 *



 * @param block A pointer to the global memory block.



 * @param pixel_data The 32-bit unsigned integer containing encoded pixel data.



 */



void WritePixelData(global unsigned char* block, unsigned int pixel_data) {



	block[4] |= pixel_data >> 24;           // Most significant byte of pixel_data.



	block[5] |= (pixel_data >> 16) & 0xff; // Second most significant byte.



	block[6] |= (pixel_data >> 8) & 0xff;  // Third most significant byte.



	block[7] |= pixel_data & 0xff;         // Least significant byte.



}







/**



 * @brief Writes a 'flip' bit into the output block.



 *



 * This function sets or clears the least significant bit of the 3rd byte



 * (`block[3]`) to indicate whether the block's orientation should be



 * "flipped" (e.g., along a diagonal or axis) during decompression.



 *



 * @param block A pointer to the global memory block.



 * @param flip The flip flag (0 or 1).



 */



void WriteFlip(global unsigned char* block, char flip) {



	block[3] &= ~0x01; // Clears the LSB.



	block[3] |= (unsigned char) flip; // Sets the LSB to the value of `flip`.



}







/**



 * @brief Writes a 'diff' bit into the output block.



 *



 * This function sets or clears the second least significant bit of the 3rd byte



 * (`block[3]`) to indicate whether differential encoding was used for the colors



 * in the block.



 *



 * @param block A pointer to the global memory block.



 * @param diff The differential encoding flag (0 or 1).



 */



void WriteDiff(global unsigned char* block, char diff) {



	block[3] &= ~0x02; // Clears the second LSB.



	block[3] |= (unsigned char) (diff) << 1; // Sets the second LSB to the value of `diff`.



}







/**



 * @brief Extracts a 4x4 pixel block (16 pixels) from a source image.



 *



 * This inline function copies a 4x4 block of 4-byte pixels (BGRA) from a



 * source image row by row into a destination buffer. It accounts for the



 * `width` parameter to correctly stride through the source image.



 *



 * @param dst A pointer to the destination buffer where the 4x4 block will be stored.



 * @param src A pointer to the starting pixel of the 4x4 block in the source image.



 * @param width The width of the source image in pixels.



 */



inline void ExtractBlock(global unsigned char* dst, const unsigned char* src, int width) {



	int i,j;







	// Iterates through 4 rows of the 4x4 block.



	for (i = 0; i < 4; ++i) {



		int index = i * 4 * 4; // Calculate starting index for the current row in dst. (4 pixels/row * 4 bytes/pixel)



		// Copies 16 bytes (4 pixels) for the current row.



		for (j = 0; j < 16; ++j) {



			dst[index + j] = src[j];



		}



		src += width * 4; // Advance source pointer to the next row of the 4x4 block. (width pixels * 4 bytes/pixel)



	}



}











/**



 * @brief Creates a `Color` object from floating-point BGR components, quantizing to 4 bits per channel.



 *



 * This function takes float BGR values (presumably 0-255), rounds them to 4-bit



 * precision, and then expands these 4-bit values back to 8-bit for each channel



 * by replicating the 4 bits (e.g., 0bAAAA -> 0bAAAAAAAA). The alpha channel is



 * set to a constant `0x44`.



 *



 * @param bgr An array of three floats representing Blue, Green, and Red components.



 * @return A `Color` structure with 4-bit quantized and expanded BGRA channels.



 */



inline Color makeColor444(const float* bgr) {



	unsigned char b4 = round_to_4_bits(bgr[0]);



	unsigned char g4 = round_to_4_bits(bgr[1]);











	unsigned char r4 = round_to_4_bits(bgr[2]);



	Color bgr444;



	bgr444.channels.b = (b4 << 4) | b4; // Expand 4-bit to 8-bit (e.g., 0xA -> 0xAA)



	bgr444.channels.g = (g4 << 4) | g4; // Expand 4-bit to 8-bit



	bgr444.channels.r = (r4 << 4) | r4; // Expand 4-bit to 8-bit



	



	bgr444.channels.a = 0x44; // Constant alpha value.



	return bgr444;



}











/**



 * @brief Creates a `Color` object from floating-point BGR components, quantizing to 5 bits per channel.



 *



 * This function takes float BGR values (presumably 0-255), rounds them to 5-bit



 * precision. However, the current implementation appears to incorrectly convert



 * the 5-bit rounded values (`b5`, `g5`, `r5`) into boolean-like values `(> 2)`



 * instead of a proper 8-bit representation or packing. The alpha channel is



 * set to a constant `0x55`.



 *



 * @param bgr An array of three floats representing Blue, Green, and Red components.



 * @return A `Color` structure with 5-bit quantized (and potentially erroneous) BGRA channels.



 */



inline Color makeColor555(const float* bgr) {



	unsigned char b5 = round_to_5_bits(bgr[0]);



	unsigned char g5 = round_to_5_bits(bgr[1]);











	unsigned char r5 = round_to_5_bits(bgr[2]);



	Color bgr555;



	// @note The following lines (b5 > 2, g5 > 2, r5 > 2) seem to be incorrect logic



	// for converting a 5-bit value back to an 8-bit channel or packing it.



	// They would result in channels being 0 or 1.



	bgr555.channels.b = (b5 > 2);



	bgr555.channels.g = (g5 > 2);



	bgr555.channels.r = (r5 > 2);



	



	bgr555.channels.a = 0x55; // Constant alpha value.



	return bgr555;



}



	



/**



 * @brief Computes the average BGR color for a given array of Color objects.



 *



 * This function sums the B, G, and R channel values of 8 input `Color` structures



 * and then divides each sum by 8 to get the average. The results are stored



 * in a float array.



 *



 * @param src A pointer to an array of 8 `Color` structures.



 * @param avg_color A pointer to a float array of size 3 to store the average BGR components.



 */



void getAverageColor(const Color* src, float* avg_color)



{



	unsigned int sum_b = 0, sum_g = 0, sum_r = 0, i;



	



	// Sums the B, G, R components of 8 colors.



	for (i = 0; i < 8; ++i) {



		sum_b += src[i].channels.b;



		sum_g += src[i].channels.g;



		sum_r += src[i].channels.r;



	}



	



	const float kInv8 = 1.0f / 8.0f; // Pre-calculated inverse for division.



	// Calculates the average B, G, R components.



	avg_color[0] = (float) sum_b * kInv8;











	avg_color[1] = (float) sum_g * kInv8;



	avg_color[2] = (float) sum_r * kInv8;



}







/**



 * @brief Computes the best luminance codeword table and pixel modulation data for a color sub-block.



 *



 * This function iterates through a set of pre-defined codeword tables and, for each table,



 * tries to find the best luminance modifier for each pixel in the input `src` sub-block.



 * It minimizes the color error between the original pixels and the candidate colors



 * derived from the base color and luminance modifiers. The best table and corresponding



 * pixel modulation data are then written to the output `block`.



 *



 * @param block A pointer to the global memory output block.



 * @param src A pointer to the array of 8 `Color` structures representing the sub-block pixels.



 * @param base The base color for calculating candidate colors.



 * @param sub_block_id The ID of the current sub-block being processed (0 or 1).



 * @param idx_to_num_tab A pointer to a constant table for remapping pixel indices.



 * @param threshold An error threshold for early exit optimization.



 * @return The minimum total error achieved for the best codeword table.



 */



unsigned long computeLuminance(global unsigned char* block,



						   const Color* src,



						   const Color base,



						   int sub_block_id,



						   __constant unsigned char* idx_to_num_tab,



						   unsigned long threshold)



{



	unsigned int best_tbl_err = threshold;   // Stores the minimum error found for a codeword table.



	unsigned char best_tbl_idx = 0;          // Stores the index of the best codeword table.



	unsigned char best_mod_idx[8][8];        // Stores the best modulation index for each pixel for each table.



	unsigned int tbl_idx, i, mod_idx;







	



	// Iterates through all 8 possible codeword tables.



	for (tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {



		



		// Generates the 4 candidate colors for the current codeword table



		// by applying luminance offsets to the `base` color.



		Color candidate_color[4];



		for (mod_idx = 0; mod_idx < 4; ++mod_idx) {



			short lum = g_codeword_tables[tbl_idx][mod_idx]; // Get luminance offset from global table.



			candidate_color[mod_idx] = makeColor(base, lum); // Create candidate color.



		}



		



		unsigned int tbl_err = 0; // Accumulates error for the current table.



		



		// For each pixel in the sub-block, find the best matching candidate color.



		for (i = 0; i < 8; ++i) {



			unsigned int best_mod_err = threshold; // Minimum error for the current pixel.



			// Iterate through the 4 candidate colors.



			for (mod_idx = 0; mod_idx < 4; ++mod_idx) {



				const Color color = candidate_color[mod_idx];



				



				// Calculate color error between source pixel and candidate color.



				unsigned int mod_err = getColorError(src[i], color);



				if (mod_err < best_mod_err) {



					best_mod_idx[tbl_idx][i] = mod_idx; // Store best modulation index for this pixel.



					best_mod_err = mod_err;



					



					if (mod_err == 0)



						break;  // Perfect match, no need to check other mods for this pixel.



				}



			}



			



			tbl_err += best_mod_err; // Add best pixel error to total table error.



			if (tbl_err > best_tbl_err)



				break;  // Early exit: current table is already worse than the best found.



		}



		



		// If current table has a lower total error, update best table.



		if (tbl_err < best_tbl_err) {



			best_tbl_err = tbl_err;



			best_tbl_idx = tbl_idx;



			



			if (tbl_err == 0)



				break;  // Perfect table found, no need to check others.



		}



	}







	// Write the index of the best codeword table to the output block.



	WriteCodewordTable(block, sub_block_id, best_tbl_idx);







	unsigned int pix_data = 0; // Stores the encoded pixel modulation data.







	// For each pixel in the sub-block, encode its best modulation index.



	for (i = 0; i < 8; ++i) {



		unsigned char mod_idx = best_mod_idx[best_tbl_idx][i]; // Get best modulation index.



		unsigned char pix_idx = g_mod_to_pix[mod_idx];        // Map to pixel index.



		



		// Extract LSB and MSB of the mapped pixel index.



		unsigned int lsb = pix_idx & 0x1;



		unsigned int msb = pix_idx >> 1;



		



		// Determine the final texel number using the provided `idx_to_num_tab`.



		int texel_num = idx_to_num_tab[i];



		// Pack the MSB and LSB into the `pix_data` at specific bit positions.



		pix_data |= msb << (texel_num + 16);



		pix_data |= lsb << (texel_num);



	}







	// Write the accumulated pixel modulation data to the output block.



	WritePixelData(block, pix_data);







	return best_tbl_err;



}











/**



 * @brief Attempts to compress a block assuming all pixels are a solid (uniform) color.



 *



 * This function checks if all 16 pixels in the input `src` block are identical.



 * If they are, it compresses the block as a solid color block using a simplified



 * encoding, typically more efficient for uniform regions. It uses a 555-color



 * representation and finds the best luminance table to represent the single color.



 *



 * @param dst A pointer to the global memory destination block (8 bytes).



 * @param src A pointer to an array of 16 `Color` structures (the input block pixels).



 * @param error A pointer to an unsigned long to store the computed compression error.



 * @return True if the block was successfully compressed as solid, False otherwise.



 */



bool tryCompressSolidBlock(global unsigned char* dst,



						   const Color* src,



						   unsigned long* error)



{



	unsigned int i, j;



	unsigned int tbl_idx;



	unsigned int mod_idx;







	// Check if all pixels in the 4x4 block are identical.



	for (i = 1; i < 16; ++i) {



		if (src[i].bits != src[0].bits)



			return false; // Not a solid block.



	}



	



	// Initialize the destination block with zeros.



	dst[0] = 0;



	dst[1] = 0;



	dst[2] = 0;



	dst[3] = 0;



	dst[4] = 0;



	dst[5] = 0;



	dst[6] = 0;



	dst[7] = 0;



	



	// Convert the solid color to float for 555 conversion.



	float src_color_float[3] = {(float)src->channels.b,



		(float)src->channels.g,



		(float)src->channels.r};



	Color base = makeColor555(src_color_float); // Base color using 5-bit precision.



	



	WriteDiff(dst, true);  // Mark as differential encoding (since base color is explicitly encoded).



	WriteFlip(dst, false); // No flip for solid blocks.



	WriteColors555(dst, base, base); // Write the same base color twice.



	



	unsigned char best_tbl_idx = 0;   // Best codeword table index.



	unsigned char best_mod_idx = 0;   // Best modulation index.



	unsigned int best_mod_err = 4294967295; // Initialize with max error.



	



	// Iterate through codeword tables and modulation indices to find the best match for the solid color.



	for (tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {



		



		for ( mod_idx = 0; mod_idx < 4; ++mod_idx) {



			short lum = g_codeword_tables[tbl_idx][mod_idx]; // Get luminance offset.



			const Color color = makeColor(base, lum);        // Create candidate color.



			



			unsigned int mod_err = getColorError(*src, color); // Calculate error.



			if (mod_err < best_mod_err) {



				best_tbl_idx = tbl_idx;



				best_mod_idx = mod_idx;



				best_mod_err = mod_err;



				



				if (mod_err == 0)



					break;  // Perfect match.



			}



		}



		



		if (best_mod_err == 0)



			break; // Perfect match found, no need to check other tables.



	}



	



	WriteCodewordTable(dst, 0, best_tbl_idx); // Write best table for sub-block 0.



	WriteCodewordTable(dst, 1, best_tbl_idx); // Write best table for sub-block 1.



	



	// Encode the pixel modulation data for a solid block (all pixels use the same best modulation index).



	unsigned char pix_idx = g_mod_to_pix[best_mod_idx];



	unsigned int lsb = pix_idx & 0x1;



	unsigned int msb = pix_idx >> 1;



	



	unsigned int pix_data = 0;



	for (i = 0; i < 2; ++i) { // This loop structure seems to encode 16 pixels.



		for (j = 0; j < 8; ++j) {



			



			int texel_num = g_idx_to_num[i][j];



			pix_data |= msb << (texel_num + 16); // Pack MSB.



			pix_data |= lsb << (texel_num);      // Pack LSB.



		}



	}



	



	WritePixelData(dst, pix_data); // Write the packed pixel data.



	*error = 16 * best_mod_err;    // Total error is 16 times the best error for a single pixel.



	return true;



}







/**



 * @brief Compresses a single 4x4 color block using various encoding strategies.



 *



 * This function attempts to compress a block by first checking for solid color blocks.



 * If not solid, it divides the 4x4 block into 2x4 sub-blocks and analyzes their average



 * colors. It then decides whether to use differential encoding or 444 encoding, and



 * whether to "flip" the sub-block orientation, based on minimizing a perceived error metric.



 * It then calls `computeLuminance` for each sub-block to determine final pixel data.



 *



 * @param dst A pointer to the global memory destination block (8 bytes).



 * @param ver_src A pointer to the vertically oriented 4x4 pixel source data.



 * @param hor_src A pointer to the horizontally oriented 4x4 pixel source data.



 * @param threshold An error threshold for early exit optimizations.



 * @return The total compression error for the block.



 */



unsigned long compressBlock(global unsigned char* dst,



												   const Color* ver_src,



												   const Color* hor_src,



												   unsigned long threshold)



{



	unsigned long solid_error = 0;



	unsigned int i, j;



	unsigned int light_idx;







	// First, try to compress as a solid color block. If successful, return its error.



	if (tryCompressSolidBlock(dst, ver_src, &solid_error)) {



		return solid_error;



	}



	



	// Pointers to the 4 sub-blocks (each 8 pixels) based on vertical and horizontal partitioning.



	const Color* sub_block_src[4] = {ver_src, ver_src + 8, hor_src, hor_src + 8};



	



	Color sub_block_avg[4];        // Stores average colors for each sub-block.



	bool use_differential[2] = {true, true}; // Flags to indicate if differential encoding is used for a pair of sub-blocks.



	



	



	// Calculates average colors for pairs of sub-blocks and decides on differential encoding.



	// This loop processes two pairs of sub-blocks (i.e., (0,1) and (2,3)).



	for (i = 0, j = 1; i < 4; i += 2, j += 2) {



		float avg_color_0[3];



		getAverageColor(sub_block_src[i], avg_color_0); // Get average for sub-block `i`.











		Color avg_color_555_0 = makeColor555(avg_color_0); // Quantize to 5-bit precision.



		



		float avg_color_1[3];



		getAverageColor(sub_block_src[j], avg_color_1); // Get average for sub-block `j`.



		Color avg_color_555_1 = makeColor555(avg_color_1); // Quantize to 5-bit precision.



		



		// For each color component (R, G, B), check if differential encoding is suitable.



		for (light_idx = 0; light_idx < 3; ++light_idx) {



			int u = avg_color_555_0.components[light_idx] >> 3; // Get 5-bit component.



			int v = avg_color_555_1.components[light_idx] >> 3; // Get 5-bit component.



			



			int component_diff = v - u;



			// If the difference between components is too large, differential encoding is not used.



			if (component_diff < -3 || component_diff > 3) {



				use_differential[i / 2] = false; // Disable differential for this pair.



				sub_block_avg[i] = makeColor444(avg_color_0); // Use 444 encoding.











				sub_block_avg[j] = makeColor444(avg_color_1); // Use 444 encoding.



			} else {



				sub_block_avg[i] = avg_color_555_0; // Use 555 encoding.



				sub_block_avg[j] = avg_color_555_1; // Use 555 encoding.



			}



		}



	}



	



	// Calculate the error if each sub-block were represented by its average color.



	// This helps in deciding the 'flip' state.



	unsigned int sub_block_err[4] = {0};



	for (i = 0; i < 4; ++i) {



		for (j = 0; j < 8; ++j) {



			sub_block_err[i] += getColorError(sub_block_avg[i], sub_block_src[i][j]);



		}



	}



	



	// Determine the 'flip' state based on which partition (vertical or horizontal)



	// results in less error when using average colors.



	char flip =



	sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1];



	



	// Initialize the destination block with zeros.



	dst[0] = 0;



	dst[1] = 0;



	dst[2] = 0;



	dst[3] = 0;



	dst[4] = 0;



	dst[5] = 0;



	dst[6] = 0;



	dst[7] = 0;



	



	// Write the 'diff' and 'flip' flags to the block header.



	WriteDiff(dst, use_differential[!!flip]);



	WriteFlip(dst, flip);



	



	// Determine which sub-block averages to use based on the 'flip' state.



	unsigned char sub_block_off_0 = flip ? 2 : 0;



	unsigned char sub_block_off_1 = sub_block_off_0 + 1;



	



	// Write the base colors for the sub-blocks, either 555 (differential) or 444 encoding.



	if (use_differential[!!flip]) {



		WriteColors555(dst, sub_block_avg[sub_block_off_0],



					   sub_block_avg[sub_block_off_1]);



	} else {



		WriteColors444(dst, sub_block_avg[sub_block_off_0],



					   sub_block_avg[sub_block_off_1]);



	}



	



	unsigned long lumi_error1 = 0, lumi_error2 = 0;



	



	// Compute luminance and pixel data for the first sub-block.



	lumi_error1 = computeLuminance(dst, sub_block_src[sub_block_off_0],



								   sub_block_avg[sub_block_off_0], 0,



								   g_idx_to_num[sub_block_off_0],



								   threshold);



	



	// Compute luminance and pixel data for the second sub-block.



	lumi_error2 = computeLuminance(dst, sub_block_src[sub_block_off_1],



								   sub_block_avg[sub_block_off_1], 1,



								   g_idx_to_num[sub_block_off_1],



								   threshold);



	



	return lumi_error1 + lumi_error2; // Return total error for the block.



}







/**



 * @brief OpenCL kernel entry point for parallel image block compression.



 *



 * This kernel processes an image block by block. Each global work-item is



 * responsible for compressing one 4x4 image block. It extracts the block data,



 * prepares it for compression (e.g., in both vertical and horizontal orientations),



 * and then calls `compressBlock` to perform the actual compression.



 * The total compression error is accumulated atomically.



 *



 * @param width The width of the input image in pixels.



 * @param height The height of the input image in pixels.



 * @param compress_error A global memory pointer to an unsigned int where the total



 *                       compression error will be atomically accumulated.



 * @param src A global memory pointer to the source image data (BGRA 8-bit per channel).



 * @param dst A global memory pointer to the destination buffer for compressed blocks.



 */



__kernel void compress_kernel(int width, int height,



                              global unsigned int *compress_error,



															global unsigned char *src,



                              global unsigned char *dst)



{



 	unsigned int num_rows = height / 4, num_columns = width / 4; // Calculate number of 4x4 blocks in rows/columns.



	unsigned int gid = get_global_id(0);                         // Get the global ID of the current work-item.



	unsigned int i, j;



  unsigned int row_index = gid / num_columns;              // Calculate the row index of the current 4x4 block.



  unsigned int column_index = gid - row_index * num_columns; // Calculate the column index of the current 4x4 block.











	unsigned int src_index = (row_index * width + column_index * 4) * 4; // Calculate starting byte index for src block.



  unsigned int dst_index = gid * 8;                                  // Calculate starting byte index for dst block. (8 bytes per compressed block)



  



  Color ver_blocks[16], hor_blocks[16]; // Buffers to store 4x4 block data in vertical and horizontal orientations.



  global Color *row_start; // Pointer to iterate through source image rows.







  row_start = (global Color *) &(src[src_index]); // Initialize row_start to the beginning of the current block.







	// Extract pixels for 'ver_blocks' (vertical orientation). This loop extracts 2 pixels per row for 4 rows.



	for (i = 0; i <= 6; i+=2) {











      ver_blocks[i].bits = row_start->bits;    // First pixel.



      ver_blocks[i+1].bits = (row_start+1)->bits; // Second pixel.



      row_start+=width; // Move to the next row in the source image.



	}







  row_start = (global Color *) &(src[src_index]); // Reset row_start to the beginning of the block.







	// Extract remaining pixels for 'ver_blocks' (vertical orientation).



	for (i = 8; i <= 14; i+=2) {











      ver_blocks[i].bits = (row_start+2)->bits; // Third pixel.



      ver_blocks[i+1].bits = (row_start+3)->bits; // Fourth pixel.



      row_start+=width; // Move to the next row.



	}







  row_start = (global Color *) &(src[src_index]); // Reset row_start to the beginning of the block.







  // Extract pixels for 'hor_blocks' (horizontal orientation). This loop extracts 4 pixels per row for 4 rows.



  for (i = 0; i <= 12; i+=4) {











    hor_blocks[i].bits = row_start->bits;



    hor_blocks[i+1].bits = (row_start+1)->bits;



    hor_blocks[i+2].bits = (row_start+2)->bits;



    hor_blocks[i+3].bits = (row_start+3)->bits;







    row_start+=width; // Move to the next row.



  }







  // Atomically add the compression error of the current block to the total error.



  atomic_add(compress_error, compressBlock(&dst[dst_index], ver_blocks, hor_blocks, 4294967295));



}







/**



 * @brief Host-side C++ code to manage OpenCL context, kernel compilation, and execution.



 * @class TextureCompressor



 * @brief Manages the OpenCL environment for texture compression.



 *



 * This class provides an interface for initializing the OpenCL platform,



 * compiling and executing the `compress_kernel`, and managing memory buffers.



 * It abstracts the complexities of OpenCL API calls.



 */



#include "compress.hpp"











using namespace std;







/**



 * @brief Constructor for TextureCompressor.



 *



 * Initializes the OpenCL environment by discovering platforms and devices,



 * creating an OpenCL context and command queue. It handles error checking



 * at each step and exits if initialization fails.



 */



TextureCompressor::TextureCompressor() {



  cl_uint platforms; // Number of OpenCL platforms found.



  cl_int rc;         // Return code for OpenCL API calls.







  // Allocate memory for platform IDs.



  platform_ids = (cl_platform_id *) malloc(sizeof(cl_platform_id) * 2);







  // Get OpenCL platform IDs.



  rc = clGetPlatformIDs(2, platform_ids, &platforms);



  if (rc != CL_SUCCESS) {



    printf("platform error\n");



    exit(1); // Exit on error.



  }







  // Get a GPU device ID for the first platform.



  rc = clGetDeviceIDs(platform_ids[0], CL_DEVICE_TYPE_GPU, 1, &device, NULL);



  if (rc != CL_SUCCESS) {



    printf("device id error\n");



    exit(1); // Exit on error.



  }







  // Create an OpenCL context.



  context = clCreateContext(0, 1, &device, NULL, NULL, &rc);



  if (rc != CL_SUCCESS) {



    printf("context error\n");



    exit(1); // Exit on error.



  }







  // Create an OpenCL command queue.



  command_queue = clCreateCommandQueue(context, device, 0, &rc);



  if (rc != CL_SUCCESS) {



    printf("queue error\n");



    exit(1); // Exit on error.



  }



}







/**



 * @brief Destructor for TextureCompressor.



 *



 * Releases allocated memory for platform IDs and the OpenCL context.



 * Ensures proper cleanup of OpenCL resources.



 */



TextureCompressor::~TextureCompressor() {



  free(platform_ids);      // Free memory for platform IDs.



  clReleaseContext(context); // Release the OpenCL context.



}







/**



 * @brief Compresses an input image using the OpenCL kernel.



 *



 * This method loads the OpenCL kernel source from "compress.cl",



 * compiles it, creates OpenCL memory buffers for source, destination,



 * and error accumulation, sets kernel arguments, enqueues the kernel for execution,



 * and reads back the compressed image data and total compression error.



 *



 * @param src A pointer to the source image data (uint8_t array, assumed BGRA 8-bit).



 * @param dst A pointer to the destination buffer for the compressed image data.



 * @param width The width of the source image in pixels.



 * @param height The height of the source image in pixels.



 * @return The total compression error accumulated by the kernel.



 */



unsigned long TextureCompressor::compress(const uint8_t* src,



									  uint8_t* dst,



									  int width,



									  int height)



{



	size_t dimension = (width / 4) * (height / 4); // Calculate the total number of 4x4 blocks to process.



  char *kernel_source = NULL; // Buffer to store kernel source code.



  FILE *kernel_fp = NULL;     // File pointer for kernel source.



  char line_buf[300];         // Buffer for reading lines from kernel file.



  unsigned int num_of_bytes = 0; // Accumulator for total bytes of kernel source.



  cl_int rc;                  // Return code for OpenCL API calls.



  cl_mem cl_src, cl_dst, cl_compress_err; // OpenCL memory objects for buffers.



  unsigned int compress_error; // Variable to hold the total compression error read from device.







  // Open the kernel file to determine its size.



  kernel_fp = fopen("compress.cl", "r");



  if (kernel_fp == NULL) {



    fprintf(stderr, "File not found\n");



    exit(1); // Exit if kernel file not found.



  }







  // Calculate the total number of bytes in the kernel source file.



  while(fgets(line_buf, 300, kernel_fp) != NULL) {



    num_of_bytes += strlen(line_buf);



  }







  // Allocate memory for the kernel source code.



  kernel_source = (char *) malloc (num_of_bytes + 1); // +1 for null terminator.



  strcpy(kernel_source, ""); // Initialize to empty string.



  fclose(kernel_fp); // Close the file after sizing.







  // Reopen the kernel file to read its content.



  kernel_fp = fopen("compress.cl", "r");



  if (kernel_fp == NULL) {



    fprintf(stderr, "File not found\n");



    exit(1); // Exit if kernel file not found.



  }







  // Read the entire kernel source into the allocated buffer.



  while(fgets(line_buf, 300, kernel_fp) != NULL) {



    strcat(kernel_source, line_buf);



  }







  fclose(kernel_fp); // Close the kernel file.







  // Create an OpenCL program object from the kernel source code.



  program = clCreateProgramWithSource(context,



            1,



            (const char **) &kernel_source,



            NULL,



            &rc);



  if (rc != CL_SUCCESS) {



    printf("create program error\n");



    exit(1); // Exit on error.



  }







  // Build (compile) the OpenCL program for the device.



  clBuildProgram(program, 1, &device, NULL, NULL, NULL);



  // Create an OpenCL kernel object from the compiled program.



  kernel = clCreateKernel(program,



            "compress_kernel", // Name of the kernel function.



            &rc);







  // Create OpenCL buffer for source image data.



  cl_src = clCreateBuffer(context,



            CL_MEM_COPY_HOST_PTR, // Copy data from host pointer.



            4*width*height,       // Size in bytes (width * height * 4 bytes/pixel).



            (void *) src,         // Host pointer to source data.



            &rc);



  if (rc != CL_SUCCESS) {



    printf("src buffer error\n");



    exit(1); // Exit on error.



  }



  // Create OpenCL buffer for compressed destination data.



  cl_dst = clCreateBuffer(context,



            CL_MEM_READ_WRITE,   // Read/write access on device.



            width*height*8/16, // Assuming 8 bytes per 4x4 block, and total blocks (width/4 * height/4)



                                 // Simplified: 8 bytes per 16 pixels. So (width*height/16) * 8



                                 // Or width*height*4/8 bytes, which seems to imply 4 bits per pixel overall.



                                 // If (width*height/16) blocks * 8 bytes/block = width*height/2 bytes.



            (void *) NULL,       // No initial host data for destination.



            &rc);



  if (rc != CL_SUCCESS) {



    printf("dst buffer error\n");



    exit(1); // Exit on error.



  }



  // Create OpenCL buffer for accumulating compression error.



  cl_compress_err = clCreateBuffer(context,



                    CL_MEM_READ_WRITE,    // Read/write access on device.



                    sizeof(unsigned int), // Size of a single unsigned int.



                    (void *) NULL,        // No initial host data.



                    &rc);



  if (rc != CL_SUCCESS) {



    printf("error buffer error\n");



    exit(1); // Exit on error.



  }



  // Set kernel arguments.



  clSetKernelArg(kernel, 0, sizeof(int), &width);



  clSetKernelArg(kernel, 1, sizeof(int), &height);



  clSetKernelArg(kernel, 2, sizeof(cl_mem), &cl_compress_err);



  clSetKernelArg(kernel, 3, sizeof(cl_mem), &cl_src);











  clSetKernelArg(kernel, 4, sizeof(cl_mem), &cl_dst);







  // Enqueue the OpenCL kernel for execution.



  rc = clEnqueueNDRangeKernel(command_queue,



                              kernel,



                              1,        // One-dimensional global work-size.



                              NULL,



                              &dimension, // Total number of global work-items (blocks).



                              NULL,     // Local work-size (automatically determined).



                              0,



                              NULL,



                              NULL);



  if (rc != CL_SUCCESS) {



    printf("enqueue error\n");



    exit(1); // Exit on error.



  }







  // Read the compressed image data from the device buffer back to host memory.



  rc = clEnqueueReadBuffer(this->command_queue,



                            cl_dst,



                            CL_TRUE, // Blocking read.



                            0,



                            width*height*4/8, // Size of compressed data.



                            dst,



                            0,



                            NULL,



                            NULL);



  if (rc != CL_SUCCESS) {



    printf("read dst error\n");



    exit(1); // Exit on error.



  }







  // Read the total compression error from the device buffer back to host memory.



  rc = clEnqueueReadBuffer(command_queue,



                            cl_compress_err,



                            CL_TRUE, // Blocking read.



                            0,



                            sizeof(unsigned int),



                            &compress_error,



                            0,



                            NULL,



                            NULL);



  if (rc != CL_SUCCESS) {



    printf("read error buffer error\n");



    exit(1); // Exit on error.



  }







  free(kernel_source); // Free allocated kernel source memory.



	return compress_error; // Return the total compression error.



}




