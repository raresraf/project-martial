/**
 * @file compress.cl
 * @brief This file contains OpenCL kernel code for image compression and
 *        a C++ host-side class for managing the compression process.
 */
// Original content starts here.
typedef union Tag {
	struct BgraColorType {
		unsigned char b;
		unsigned char g;
		unsigned char r;
		unsigned char a;
	} channels;
	unsigned char components[4];
	unsigned int bits;
} Color;

// Pre-condition: val, min, and max are integers.
// Invariant: Ensures val is within the [min, max] range.
int clamp1(int val, int min, int max) {
	return val  max ? max : val);
}

// Pre-condition: val, min, and max are unsigned integers.
// Invariant: Ensures val is within the [min, max] range.
unsigned int clamp2(unsigned int val, unsigned int min, unsigned int max) {
	return val  max ? max : val);
}

// Pre-condition: val is a float representing a color component (0-255).
// Invariant: Converts a float value to an unsigned char (5-bit representation).
unsigned char round_to_5_bits(float val) {
	return (unsigned char) clamp2(val * 31.0f / 255.0f + 0.5f, 0, 31);
}

// Pre-condition: val is a float representing a color component (0-255).
// Invariant: Converts a float value to an unsigned char (4-bit representation).
unsigned char round_to_4_bits(float val) {
	return (unsigned char) clamp2(val * 15.0f / 255.0f + 0.5f, 0, 15);
}

__constant short g_codeword_tables[8][4] = {
	{-8, -2, 2, 8},
	{-17, -5, 5, 17},
	{-29, -9, 9, 29},
	{-42, -13, 13, 42},
	{-60, -18, 18, 60},


	{-80, -24, 24, 80},
	{-106, -33, 33, 106},
	{-183, -47, 47, 183}};

__constant unsigned char g_mod_to_pix[4] = {3, 2, 0, 1};

__constant unsigned char g_idx_to_num[4][8] = {
	{0, 4, 1, 5, 2, 6, 3, 7},        
	{8, 12, 9, 13, 10, 14, 11, 15},  
	{0, 4, 8, 12, 1, 5, 9, 13},      
	{2, 6, 10, 14, 3, 7, 11, 15}     
};

// Pre-condition: base is a Color struct, lum is a short luminance value.
// Invariant: Adjusts the base color by adding luminance to its RGB channels.
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

#define USE_PERCEIVED_ERROR_METRIC


// Pre-condition: u and v are valid Color structs.
// Invariant: Calculates the color error between two colors.
unsigned int getColorError(const Color u, const Color v) {
#ifdef USE_PERCEIVED_ERROR_METRIC
	float delta_b = (float) u.channels.b - v.channels.b;


	float delta_g = (float) u.channels.g - v.channels.g;
	float delta_r = (float) u.channels.r - v.channels.r;
	// Functional utility: Multiplies delta values by perceptual weights and sums them.
	return (unsigned int) (0.299f * delta_b * delta_b +
								 0.587f * delta_g * delta_g +
								 0.114f * delta_r * delta_r);
#else
	int delta_b = static_cast(u.channels.b) - v.channels.b;
	int delta_g = static_cast(u.channels.g) - v.channels.g;
	int delta_r = static_cast(u.channels.r) - v.channels.r;
	return delta_b * delta_b + delta_g * delta_g + delta_r * delta_r;
#endif
}



// Pre-condition: block is a global unsigned char pointer, color0 and color1 are Color structs.
// Invariant: Writes 4-bit color components of color0 and color1 into the block, packed.
void WriteColors444(global unsigned char* block,
									 const Color color0,
									 const Color color1) {
	
	// Functional utility: Combines the most significant 4 bits of color0.r and least significant 4 bits of color1.r
	block[0] = (color0.channels.r & 0xf0) | (color1.channels.r >> 4);
	// Functional utility: Combines the most significant 4 bits of color0.g and least significant 4 bits of color1.g
	block[1] = (color0.channels.g & 0xf0) | (color1.channels.g >> 4);
	// Functional utility: Combines the most significant 4 bits of color0.b and least significant 4 bits of color1.b
	block[2] = (color0.channels.b & 0xf0) | (color1.channels.b >> 4);
}

// Pre-condition: block is a global unsigned char pointer, color0 and color1 are Color structs.
// Invariant: Writes 5-bit color components of color0 and the difference (delta) from color1 into the block.
void WriteColors555(global unsigned char* block,
						   const Color color0,


						   const Color color1) {
	
	const unsigned char two_compl_trans_table[8] = {
		4,  
		5,  
		6,  
		7,  
		0,  
		1,  
		2,  
		3,  
	};
	
	short delta_r =
	// Functional utility: Extracts the 5-bit red component and calculates the difference.
	(short) (color1.channels.r >> 3) - (color0.channels.r >> 3);
	short delta_g =
	// Functional utility: Extracts the 5-bit green component and calculates the difference.
	(short) (color1.channels.g >> 3) - (color0.channels.g >> 3);
	short delta_b =
	// Functional utility: Extracts the 5-bit blue component and calculates the difference.
	(short) (color1.channels.b >> 3) - (color0.channels.b >> 3);
	
	
	// Functional utility: Combines the most significant 5 bits of color0.r with a transformed delta_r.
	block[0] = (color0.channels.r & 0xf8) | two_compl_trans_table[delta_r + 4];
	// Functional utility: Combines the most significant 5 bits of color0.g with a transformed delta_g.
	block[1] = (color0.channels.g & 0xf8) | two_compl_trans_table[delta_g + 4];
	// Functional utility: Combines the most significant 5 bits of color0.b with a transformed delta_b.
	block[2] = (color0.channels.b & 0xf8) | two_compl_trans_table[delta_b + 4];
}



// Pre-condition: block is a global unsigned char pointer, sub_block_id (0 or 1), table (0-7).
// Invariant: Writes the codeword table index for a sub-block into the block.
void WriteCodewordTable(global unsigned char* block,
							   unsigned char sub_block_id,
							   unsigned char table) {
	
	// Functional utility: Calculates the bit shift for the sub_block_id.
	unsigned char shift = (2 + (3 - sub_block_id * 3));
	// Functional utility: Clears the relevant bits in block[3] for the table.
	block[3] &= ~(0x07 << shift);
	// Functional utility: Sets the table bits in block[3].
	block[3] |= table << shift;


}

// Pre-condition: block is a global unsigned char pointer, pixel_data is an unsigned int.
// Invariant: Writes pixel data into bytes block[4] through block[7] using bit shifts and masks.
void WritePixelData(global unsigned char* block, unsigned int pixel_data) {
	// Functional utility: Extracts the most significant 8 bits of pixel_data.
	block[4] |= pixel_data >> 24;
	// Functional utility: Extracts the next 8 bits of pixel_data.
	block[5] |= (pixel_data >> 16) & 0xff;
	// Functional utility: Extracts the next 8 bits of pixel_data.
	block[6] |= (pixel_data >> 8) & 0xff;
	// Functional utility: Extracts the least significant 8 bits of pixel_data.
	block[7] |= pixel_data & 0xff;
}

// Pre-condition: block is a global unsigned char pointer, flip is a char (0 or 1).
// Invariant: Writes the flip flag into block[3].
void WriteFlip(global unsigned char* block, char flip) {
	// Functional utility: Clears the least significant bit of block[3].
	block[3] &= ~0x01;
	// Functional utility: Sets the least significant bit of block[3] to the flip value.
	block[3] |= (unsigned char) flip;
}

// Pre-condition: block is a global unsigned char pointer, diff is a char (0 or 1).
// Invariant: Writes the differential flag into block[3].
void WriteDiff(global unsigned char* block, char diff) {
	// Functional utility: Clears the second least significant bit of block[3].
	block[3] &= ~0x02;
	// Functional utility: Sets the second least significant bit of block[3] to the diff value.
	block[3] |= (unsigned char) (diff) << 1;
}

// Pre-condition: dst and src are global unsigned char pointers, width is an int.
// Invariant: Extracts a 4x4 block of data from src into dst, handling pitch.
inline void ExtractBlock(global unsigned char* dst, const unsigned char* src, int width) {
	int i,j;

	// Pre-condition: i is the row index (0-3).
	// Invariant: Iterates through each row of the 4x4 block.
	for (i = 0; i < 4; ++i) {
		// Functional utility: Calculates the starting index for the current row in the destination.
		int index = i * 4 * 4;
		// Pre-condition: j is the column index (0-15 for 4 channels * 4 columns).
		// Invariant: Copies 16 bytes (4 pixels) for the current row.
		for (j = 0; j < 16; ++j) {
			// Functional utility: Pointer arithmetic to copy byte from source to destination.
			dst[index + j] = src[j];
		}


		// Functional utility: Advances the source pointer to the next row in memory.
		src += width * 4;
	}
}





// Pre-condition: bgr is a float array containing blue, green, red components.
// Invariant: Creates a Color struct with 4-bit per channel color data.
inline Color makeColor444(const float* bgr) {
	unsigned char b4 = round_to_4_bits(bgr[0]);
	unsigned char g4 = round_to_4_bits(bgr[1]);
	unsigned char r4 = round_to_4_bits(bgr[2]);
	Color bgr444;
	// Functional utility: Duplicates the 4-bit value to fill an 8-bit channel (e.g., 0bAAAA becomes 0bAAAAAAAA).
	bgr444.channels.b = (b4 << 4) | b4;
	// Functional utility: Duplicates the 4-bit value to fill an 8-bit channel.
	bgr444.channels.g = (g4 << 4) | g4;


	// Functional utility: Duplicates the 4-bit value to fill an 8-bit channel.
	bgr444.channels.r = (r4 << 4) | r4;
	
	bgr444.channels.a = 0x44;
	return bgr444;
}





// Pre-condition: bgr is a float array containing blue, green, red components.
// Invariant: Creates a Color struct with 5-bit per channel color data.
inline Color makeColor555(const float* bgr) {
	unsigned char b5 = round_to_5_bits(bgr[0]);
	unsigned char g5 = round_to_5_bits(bgr[1]);
	unsigned char r5 = round_to_5_bits(bgr[2]);
	Color bgr555;
	// Functional utility: Sets the blue channel based on a comparison (likely a placeholder or error in original code 'b5 > 2').
	bgr555.channels.b = (b5 > 2);
	// Functional utility: Sets the green channel based on a comparison.
	bgr555.channels.g = (g5 > 2);


	// Functional utility: Sets the red channel based on a comparison.
	bgr555.channels.r = (r5 > 2);
	
	bgr555.channels.a = 0x55;
	return bgr555;
}
	
// Pre-condition: src is a Color array, avg_color is a float array of size 3.
// Invariant: Computes the average RGB color from the input array of 8 colors.
void getAverageColor(const Color* src, float* avg_color)
{
	unsigned int sum_b = 0, sum_g = 0, sum_r = 0, i;
	
	// Pre-condition: i is the index of the color in the src array (0-7).
	// Invariant: Accumulates the sum of red, green, and blue components.
	for (i = 0; i < 8; ++i) {
		// Functional utility: Pointer arithmetic to access color channels.
		sum_b += src[i].channels.b;
		sum_g += src[i].channels.g;
		sum_r += src[i].channels.r;
	}
	
	const float kInv8 = 1.0f / 8.0f;
	avg_color[0] = (float) sum_b * kInv8;


	avg_color[1] = (float) sum_g * kInv8;
	avg_color[2] = (float) sum_r * kInv8;
}

// Pre-condition: Parameters are valid pointers and values for luminance computation.
// Invariant: Computes the best codeword table index and modulation indices for a sub-block to minimize error.
unsigned long computeLuminance(global unsigned char* block,
						   const Color* src,
						   const Color base,
						   int sub_block_id,
						   __constant unsigned char* idx_to_num_tab,
						   unsigned long threshold)
{
	unsigned int best_tbl_err = threshold;
	unsigned char best_tbl_idx = 0;
	unsigned char best_mod_idx[8][8];  
	unsigned int tbl_idx, i, mod_idx;

	
	
	// Pre-condition: tbl_idx iterates through all 8 codeword tables.
	// Invariant: Finds the best codeword table that minimizes the error for the current sub-block.
	for (tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		
		
		Color candidate_color[4];  


		// Pre-condition: mod_idx iterates through all 4 modulation indices for the current table.
		// Invariant: Generates candidate colors by applying luminance to the base color.
		for (mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			candidate_color[mod_idx] = makeColor(base, lum);
		}
		
		unsigned int tbl_err = 0;
		
		// Pre-condition: i iterates through the 8 source colors in the sub-block.
		// Invariant: For each source color, finds the best matching candidate color from the current table.
		for (i = 0; i < 8; ++i) {
			
			
			unsigned int best_mod_err = threshold;


			// Pre-condition: mod_idx iterates through the 4 candidate colors.
			// Invariant: Finds the candidate color that minimizes error with the current source color.
			for (mod_idx = 0; mod_idx < 4; ++mod_idx) {
				const Color color = candidate_color[mod_idx];
				
				unsigned int mod_err = getColorError(src[i], color);
				// Pre-condition: mod_err is the error for the current candidate color.
				// Invariant: Updates best_mod_err and best_mod_idx if a better match is found.
				if (mod_err < best_mod_err) {
					best_mod_idx[tbl_idx][i] = mod_idx;
					best_mod_err = mod_err;
					
					// Pre-condition: mod_err is 0.
					// Invariant: If a perfect match is found, no need to check further candidates for this src color.
					if (mod_err == 0)
						break;  
				}
			}
			
			tbl_err += best_mod_err;
			// Pre-condition: tbl_err exceeds the best_tbl_err found so far.
			// Invariant: Prunes the search if the current table is already worse than the best.
			if (tbl_err > best_tbl_err)


				break;  
		}
		
		// Pre-condition: tbl_err is the total error for the current table.
		// Invariant: Updates best_tbl_err and best_tbl_idx if the current table is better.
		if (tbl_err < best_tbl_err) {
			best_tbl_err = tbl_err;
			best_tbl_idx = tbl_idx;
			
			// Pre-condition: tbl_err is 0.
			// Invariant: If a perfect table is found, no need to check further tables.
			if (tbl_err == 0)
				break;  
		}
	}

	WriteCodewordTable(block, sub_block_id, best_tbl_idx);

	unsigned int pix_data = 0;

	// Pre-condition: i iterates through the 8 colors in the sub-block.
	// Invariant: Packs the modulation indices into pix_data for storage.
	for (i = 0; i < 8; ++i) {


		unsigned char mod_idx = best_mod_idx[best_tbl_idx][i];
		// Functional utility: Maps the modulation index to a pixel index.
		unsigned char pix_idx = g_mod_to_pix[mod_idx];
		
		// Functional utility: Extracts the least significant bit of pix_idx.
		unsigned int lsb = pix_idx & 0x1;
		// Functional utility: Extracts the most significant bit of pix_idx.
		unsigned int msb = pix_idx >> 1;
		
		
		// Functional utility: Determines the texel number based on the current index.
		int texel_num = idx_to_num_tab[i];
		// Functional utility: Shifts and ORs msb into pix_data.
		pix_data |= msb << (texel_num + 16);
		// Functional utility: Shifts and ORs lsb into pix_data.
		pix_data |= lsb << (texel_num);
	}

	WritePixelData(block, pix_data);

	return best_tbl_err;
}


// Pre-condition: dst is a global unsigned char pointer, src is a Color array (16 elements), error is an unsigned long pointer.
// Invariant: Attempts to compress a solid color block. Returns true if solid, false otherwise.
bool tryCompressSolidBlock(global unsigned char* dst,
						   const Color* src,
						   unsigned long* error)
{
	unsigned int i, j;
	unsigned int tbl_idx;
	unsigned int mod_idx;

	// Pre-condition: i iterates from 1 to 15.
	// Invariant: Checks if all colors in the block are identical to the first color.
	for (i = 1; i < 16; ++i) {
		// Functional utility: Compares the raw bit representation of colors.
		if (src[i].bits != src[0].bits)
			return false;
	}
	
	
	// Functional utility: Initializes the destination block to all zeros.
	dst[0] = 0;
	dst[1] = 0;
	dst[2] = 0;
	dst[3] = 0;
	dst[4] = 0;
	dst[5] = 0;
	dst[6] = 0;
	dst[7] = 0;
	
	float src_color_float[3] = {src->channels.b,
		src->channels.g,


		src->channels.r};
	Color base = makeColor555(src_color_float);
	
	WriteDiff(dst, true);
	WriteFlip(dst, false);
	WriteColors555(dst, base, base);
	
	unsigned char best_tbl_idx = 0;
	unsigned char best_mod_idx = 0;
	unsigned int best_mod_err = 4294967295; 
	
	
	
	// Pre-condition: tbl_idx iterates through all 8 codeword tables.
	// Invariant: Finds the best codeword table and modulation index for the solid color.
	for (tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		
		


		// Pre-condition: mod_idx iterates through all 4 modulation indices for the current table.
		// Invariant: Generates a candidate color and checks its error against the solid source color.
		for ( mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			const Color color = makeColor(base, lum);
			
			unsigned int mod_err = getColorError(*src, color);
			// Pre-condition: mod_err is the error for the current candidate color.
			// Invariant: Updates best_tbl_idx, best_mod_idx, and best_mod_err if a better match is found.
			if (mod_err < best_mod_err) {
				best_tbl_idx = tbl_idx;
				best_mod_idx = mod_idx;
				best_mod_err = mod_err;
				
				// Pre-condition: mod_err is 0.
				// Invariant: If a perfect match is found, no need to check further candidates for this table.
				if (mod_err == 0)
					break;  
			}
		}
		
		// Pre-condition: best_mod_err is 0.
		// Invariant: If a perfect match for any table is found, no need to check further tables.
		if (best_mod_err == 0)
			break;
	}
	
	WriteCodewordTable(dst, 0, best_tbl_idx);
	WriteCodewordTable(dst, 1, best_tbl_idx);
	
	unsigned char pix_idx = g_mod_to_pix[best_mod_idx];
	// Functional utility: Extracts the least significant bit of pix_idx.
	unsigned int lsb = pix_idx & 0x1;
	// Functional utility: Extracts the most significant bit of pix_idx.
	unsigned int msb = pix_idx >> 1;
	
	unsigned int pix_data = 0;
	// Pre-condition: i iterates twice (for two sub-blocks conceptually).
	// Invariant: Populates pixel data for the solid block.
	for (i = 0; i < 2; ++i) {
		// Pre-condition: j iterates through 8 elements.
		// Invariant: Packs the msb and lsb into pix_data.
		for (j = 0; j < 8; ++j) {
			
			// Functional utility: Determines the texel number from the lookup table.
			int texel_num = g_idx_to_num[i][j];
			// Functional utility: Shifts and ORs msb into pix_data.
			pix_data |= msb << (texel_num + 16);


			// Functional utility: Shifts and ORs lsb into pix_data.
			pix_data |= lsb << (texel_num);
		}
	}
	
	WritePixelData(dst, pix_data);
	// Functional utility: Calculates the total error for the solid block.
	*error = 16 * best_mod_err;
	return true;
}

// Pre-condition: dst is global unsigned char pointer, ver_src and hor_src are Color arrays, threshold is an unsigned long.
// Invariant: Compresses a block using either solid block compression or a more complex differential/flip approach.
unsigned long compressBlock(global unsigned char* dst,
												   const Color* ver_src,
												   const Color* hor_src,
												   unsigned long threshold)
{
	unsigned long solid_error = 0;
	unsigned int i, j;
	unsigned int light_idx;

	// Pre-condition: tryCompressSolidBlock is called.
	// Invariant: If the block is solid, it's compressed and the error is returned.
	if (tryCompressSolidBlock(dst, ver_src, &solid_error)) {
		return solid_error;
	}
	
	const Color* sub_block_src[4] = {ver_src, ver_src + 8, hor_src, hor_src + 8};
	
	Color sub_block_avg[4];
	bool use_differential[2] = {true, true};
	
	
	
	// Pre-condition: i iterates from 0 to 2 (steps of 2), j from 1 to 3 (steps of 2).
	// Invariant: Computes average colors for sub-blocks and decides whether to use differential encoding.
	for (i = 0, j = 1; i < 4; i += 2, j += 2) {
		float avg_color_0[3];
		getAverageColor(sub_block_src[i], avg_color_0);
		Color avg_color_555_0 = makeColor555(avg_color_0);
		
		float avg_color_1[3];
		getAverageColor(sub_block_src[j], avg_color_1);
		Color avg_color_555_1 = makeColor555(avg_color_1);
		
		// Pre-condition: light_idx iterates through 0, 1, 2 (RGB components).
		// Invariant: Checks if the difference between components is within a threshold for differential encoding.
		for (light_idx = 0; light_idx < 3; ++light_idx) {
			// Functional utility: Extracts the 5-bit component value.
			int u = avg_color_555_0.components[light_idx] >> 3;
			// Functional utility: Extracts the 5-bit component value.
			int v = avg_color_555_1.components[light_idx] >> 3;
			
			int component_diff = v - u;
			// Pre-condition: Absolute difference between components is greater than 3.
			// Invariant: If difference is too large, differential encoding is not used for this sub-block pair.
			if (component_diff  3) {
				use_differential[i / 2] = false;
				sub_block_avg[i] = makeColor444(avg_color_0);


				sub_block_avg[j] = makeColor444(avg_color_1);
			} else {
				sub_block_avg[i] = avg_color_555_0;
				sub_block_avg[j] = avg_color_555_1;
			}
		}
	}
	
	
	
	
	unsigned int sub_block_err[4] = {0};
	// Pre-condition: i iterates through the 4 sub-blocks.
	// Invariant: Computes the total error for each sub-block.
	for (i = 0; i < 4; ++i) {
		// Pre-condition: j iterates through the 8 colors within a sub-block.
		// Invariant: Accumulates error for each color in the sub-block.
		for (j = 0; j < 8; ++j) {
			sub_block_err[i] += getColorError(sub_block_avg[i], sub_block_src[i][j]);
		}
	}
	
	// Pre-condition: sub_block_err array contains errors for all sub-blocks.
	// Invariant: Determines if sub-blocks should be flipped based on their combined error.
	char flip =
	sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1];
	
	
	// Functional utility: Initializes the destination block to all zeros.
	dst[0] = 0;
	dst[1] = 0;
	dst[2] = 0;
	dst[3] = 0;
	dst[4] = 0;
	dst[5] = 0;
	dst[6] = 0;
	dst[7] = 0;
	


	WriteDiff(dst, use_differential[!!flip]); // Functional utility: `!!flip` converts char to boolean.
	WriteFlip(dst, flip);
	
	// Pre-condition: flip determines the offset.
	// Invariant: Sets the starting offsets for sub-blocks based on the flip flag.
	unsigned char sub_block_off_0 = flip ? 2 : 0;
	unsigned char sub_block_off_1 = sub_block_off_0 + 1;
	
	// Pre-condition: use_differential flag for the flipped sub-block pair.
	// Invariant: Writes colors using either 5-bit (differential) or 4-bit (non-differential) encoding.
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

// Pre-condition: width and height are dimensions, compress_error, src, and dst are global pointers.
// Invariant: Main OpenCL kernel function to compress image blocks in parallel.
__kernel void compress_kernel(int width, int height,
                              global unsigned int *compress_error,
															global unsigned char *src,
                              global unsigned char *dst)
{
 	unsigned int num_rows = height / 4, num_columns = width / 4;
	unsigned int gid = get_global_id(0);
	unsigned int i, j;
  unsigned int row_index = gid / num_columns; // Functional utility: Calculates the row index for the current global ID.
  unsigned int column_index = gid - row_index * num_columns; // Functional utility: Calculates the column index for the current global ID.
	unsigned int src_index = 16*width*row_index + column_index*4*4; // Functional utility: Calculates the starting source index for the 4x4 block.
  unsigned int dst_index = gid*4*2; // Functional utility: Calculates the starting destination index for the compressed block.
  
  Color ver_blocks[16], hor_blocks[16];
  global Color *row_start;

  // Functional utility: Initializes row_start pointer to the beginning of the current block in source.
  row_start = (global Color *) &(src[src_index]);

	// Pre-condition: i iterates from 0 to 6 with a step of 2.
	// Invariant: Populates the 'ver_blocks' array with vertical block data.
	for (i = 0; i <= 6; i+=2) {
      // Functional utility: Reads color data by accessing the raw bits.
      ver_blocks[i].bits = row_start->bits;
      // Functional utility: Reads color data for the next pixel.
      ver_blocks[i+1].bits = (row_start+1)->bits;
      // Functional utility: Advances row_start to the next row in the source image.
      row_start+=width;
	}

  // Functional utility: Re-initializes row_start pointer to the beginning of the current block in source.
  row_start = (global Color *) &(src[src_index]);

	// Pre-condition: i iterates from 8 to 14 with a step of 2.
	// Invariant: Continues populating the 'ver_blocks' array with vertical block data.
	for (i = 8; i <= 14; i+=2) {
      // Functional utility: Reads color data from an offset for the next part of the vertical block.
      ver_blocks[i].bits = (row_start+2)->bits;
      // Functional utility: Reads color data from an offset for the next part of the vertical block.
      ver_blocks[i+1].bits = (row_start+3)->bits;
      // Functional utility: Advances row_start to the next row in the source image.
      row_start+=width;
	}

  // Functional utility: Re-initializes row_start pointer to the beginning of the current block in source.
  row_start = (global Color *) &(src[src_index]);

  // Pre-condition: i iterates from 0 to 12 with a step of 4.
  // Invariant: Populates the 'hor_blocks' array with horizontal block data.
  for (i = 0; i <= 12; i+=4) {
    // Functional utility: Reads color data for the current pixel.
    hor_blocks[i].bits = row_start->bits;
    // Functional utility: Reads color data for the next pixel.
    hor_blocks[i+1].bits = (row_start+1)->bits;
    // Functional utility: Reads color data for the next pixel.
    hor_blocks[i+2].bits = (row_start+2)->bits;
    // Functional utility: Reads color data for the next pixel.
    hor_blocks[i+3].bits = (row_start+3)->bits;

    // Functional utility: Advances row_start to the next row in the source image.
    row_start+=width;
  }

  // Functional utility: Atomically adds the compression error to the global error counter.
  atomic_add(compress_error, compressBlock(&dst[dst_index], ver_blocks, hor_blocks, 4294967295));
}
#include "compress.hpp"


using namespace std;

// @brief Constructor for TextureCompressor. Initializes OpenCL context, device, and command queue.
TextureCompressor::TextureCompressor() {
  cl_uint platforms;
  cl_int rc;

  // Functional utility: Allocates memory for platform IDs.
  platform_ids = (cl_platform_id *) malloc(sizeof(cl_platform_id) * 2);

  // Pre-condition: platform_ids is allocated.
  // Invariant: Retrieves OpenCL platform IDs.
  rc = clGetPlatformIDs(2, platform_ids, &platforms);
  // Pre-condition: rc is the return code from clGetPlatformIDs.
  // Invariant: Checks for platform ID retrieval errors.
  if (rc != CL_SUCCESS) {
    printf("platform error
");
    exit(1);
  }

  // Pre-condition: platform_ids[0] is a valid platform ID.
  // Invariant: Retrieves a GPU device ID for the first platform.
  rc = clGetDeviceIDs(platform_ids[0], CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  // Pre-condition: rc is the return code from clGetDeviceIDs.
  // Invariant: Checks for device ID retrieval errors.
  if (rc != CL_SUCCESS) {
    printf("device id error
");
    exit(1);
  }

  // Pre-condition: device is a valid OpenCL device.
  // Invariant: Creates an OpenCL context for the device.
  context = clCreateContext(0, 1, &device, NULL, NULL, &rc);
  // Pre-condition: rc is the return code from clCreateContext.
  // Invariant: Checks for context creation errors.
  if (rc != CL_SUCCESS) {
    printf("context error
");
    exit(1);
  }

  // Pre-condition: context and device are valid.
  // Invariant: Creates an OpenCL command queue for the context and device.
  command_queue = clCreateCommandQueue(context, device, 0, &rc);
  // Pre-condition: rc is the return code from clCreateCommandQueue.
  // Invariant: Checks for command queue creation errors.
  if (rc != CL_SUCCESS) {
    printf("queue error
");
    exit(1);
  }
}

// @brief Destructor for TextureCompressor. Releases OpenCL resources.
TextureCompressor::~TextureCompressor() {
  free(platform_ids);
  clReleaseContext(context);
}

// @brief Compresses image data using the OpenCL kernel.
// @param src Pointer to the source image data.
// @param dst Pointer to the destination buffer for compressed data.
// @param width Width of the image.
// @param height Height of the image.
// @return The total compression error.
unsigned long TextureCompressor::compress(const uint8_t* src,
									  uint8_t* dst,
									  int width,
									  int height)
{
	// Functional utility: Calculates the number of 4x4 blocks in the image.
	size_t dimension = (width / 4) * (height / 4);
  char *kernel_source = NULL;
  FILE *kernel_fp = NULL;
  char line_buf[300];
  unsigned int num_of_bytes = 0;
  cl_int rc;
  cl_mem cl_src, cl_dst, cl_compress_err;
  unsigned int compress_error;

  // Pre-condition: "compress.cl" exists.
  // Invariant: Opens the kernel source file for reading its size.
  kernel_fp = fopen("compress.cl", "r");
  // Pre-condition: kernel_fp is NULL.
  // Invariant: Handles error if kernel file is not found.
  if (kernel_fp == NULL) {
    fprintf(stderr, "File not found
");
    exit(1);
  }

  // Pre-condition: kernel_fp is open.
  // Invariant: Reads each line to determine the total size of the kernel source.
  while(fgets(line_buf, 300, kernel_fp) != NULL) {
    num_of_bytes += strlen(line_buf);
  }

  // Functional utility: Allocates memory for the kernel source.
  kernel_source = (char *) malloc (num_of_bytes);
  strcpy(kernel_source, "");
  fclose(kernel_fp);

  // Pre-condition: "compress.cl" exists.
  // Invariant: Re-opens the kernel source file to read its content.
  kernel_fp = fopen("compress.cl", "r");
  // Pre-condition: kernel_fp is NULL.
  // Invariant: Handles error if kernel file is not found.
  if (kernel_fp == NULL) {
    fprintf(stderr, "File not found
");
    exit(1);
  }

  // Pre-condition: kernel_fp is open.
  // Invariant: Reads each line and concatenates it to kernel_source.
  while(fgets(line_buf, 300, kernel_fp) != NULL) {
    strcat(kernel_source, line_buf);
  }

  fclose(kernel_fp);

  // Pre-condition: context is valid, kernel_source contains the OpenCL kernel code.
  // Invariant: Creates an OpenCL program from the source code.
  program = clCreateProgramWithSource(context,
            1,
            (const char **) &kernel_source,
            NULL,
            &rc);
  // Pre-condition: rc is the return code from clCreateProgramWithSource.
  // Invariant: Checks for program creation errors.
  if (rc != CL_SUCCESS) {
    printf("create program error
");
    exit(1);
  }

  // Pre-condition: program and device are valid.
  // Invariant: Builds (compiles) the OpenCL program for the device.
  clBuildProgram(program, 1, &device, NULL, NULL, NULL);
  // Pre-condition: program is built.
  // Invariant: Creates an OpenCL kernel from the compiled program.
  kernel = clCreateKernel(program,
            "compress_kernel",
            &rc);

  // Pre-condition: context is valid, src pointer and dimensions are valid.
  // Invariant: Creates an OpenCL buffer for the source image data, copying from host.
  cl_src = clCreateBuffer(context,
            CL_MEM_COPY_HOST_PTR,
            4*width*height,
            (void *) src,
            &rc);
  // Pre-condition: rc is the return code from clCreateBuffer.
  // Invariant: Checks for source buffer creation errors.
  if (rc != CL_SUCCESS) {
    printf("src buffer error
");
    exit(1);
  }
  // Pre-condition: context is valid, width and height are valid.
  // Invariant: Creates an OpenCL buffer for the compressed destination data.
  cl_dst = clCreateBuffer(context,
            CL_MEM_READ_WRITE,
            width*height*4/8,
            (void *) NULL,
            &rc);
  // Pre-condition: rc is the return code from clCreateBuffer.
  // Invariant: Checks for destination buffer creation errors.
  if (rc != CL_SUCCESS) {
    printf("dst buffer error
");
    exit(1);
  }
  // Pre-condition: context is valid.
  // Invariant: Creates an OpenCL buffer to store the compression error.
  cl_compress_err = clCreateBuffer(context,
                    CL_MEM_READ_WRITE,
                    sizeof(unsigned int),
                    (void *) NULL,
                    &rc);
  // Pre-condition: rc is the return code from clCreateBuffer.
  // Invariant: Checks for error buffer creation errors.
  if (rc != CL_SUCCESS) {
    printf("error buffer error
");
    exit(1);
  }
  // Pre-condition: kernel and width are valid.
  // Invariant: Sets the 'width' argument for the kernel.
  clSetKernelArg(kernel, 0, sizeof(int), &width);
  // Pre-condition: kernel and height are valid.
  // Invariant: Sets the 'height' argument for the kernel.
  clSetKernelArg(kernel, 1, sizeof(int), &height);
  // Pre-condition: kernel and cl_compress_err are valid.
  // Invariant: Sets the 'compress_error' buffer argument for the kernel.
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &cl_compress_err);
  // Pre-condition: kernel and cl_src are valid.
  // Invariant: Sets the 'src' buffer argument for the kernel.
  clSetKernelArg(kernel, 3, sizeof(cl_mem), &cl_src);
  // Pre-condition: kernel and cl_dst are valid.
  // Invariant: Sets the 'dst' buffer argument for the kernel.
  clSetKernelArg(kernel, 4, sizeof(cl_mem), &cl_dst);

  // Pre-condition: command_queue, kernel, and dimension are valid.
  // Invariant: Enqueues the OpenCL kernel for execution.
  rc = clEnqueueNDRangeKernel(command_queue,
                              kernel,
                              1,
                              NULL,
                              &dimension,
                              NULL,
                              0,
                              NULL,
                              NULL);
  // Pre-condition: rc is the return code from clEnqueueNDRangeKernel.
  // Invariant: Checks for kernel enqueue errors.
  if (rc != CL_SUCCESS) {
    printf("enqueue error
");
    exit(1);
  }

  // Pre-condition: command_queue and cl_dst are valid.
  // Invariant: Reads the compressed data back from the device to the host 'dst' buffer.
  rc = clEnqueueReadBuffer(this->command_queue,
                            cl_dst,
                            CL_TRUE,
                            0,
                            width*height*4/8, 
                            dst,
                            0,
                            NULL,
                            NULL);
  // Pre-condition: rc is the return code from clEnqueueReadBuffer.
  // Invariant: Checks for destination buffer read errors.
  if (rc != CL_SUCCESS) {
    printf("read dst error
");
    exit(1);
  }

  // Pre-condition: command_queue and cl_compress_err are valid.
  // Invariant: Reads the total compression error back from the device.
  rc = clEnqueueReadBuffer(command_queue,
                            cl_compress_err,
                            CL_TRUE,
                            0,
                            sizeof(unsigned int), 
                            &compress_error,
                            0,
                            NULL,
                            NULL);
  // Pre-condition: rc is the return code from clEnqueueReadBuffer.
  // Invariant: Checks for error buffer read errors.
  if (rc != CL_SUCCESS) {
    printf("read error buffer error
");
    exit(1);
  }

  free(kernel_source);
	return compress_error;
}
