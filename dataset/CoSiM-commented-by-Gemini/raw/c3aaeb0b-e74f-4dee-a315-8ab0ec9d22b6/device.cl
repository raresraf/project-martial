/**
 * @file device.cl
 * @brief OpenCL kernel logic and host orchestration for ETC1 texture compression.
 * Architectural Intent: Implements the Ericsson Texture Compression (ETC1) algorithm
 * optimized for GPU parallel execution using OpenCL, with a fallback C++ implementation.
 * 
 * Domain-Specific Awareness:
 * - Uses a massively parallel NDRange grid where each work-item processes a 4x4 block.
 * - Exploits memory hierarchy by caching block data in private registers.
 * - Implements differential color encoding (555 mode) and standard (444 mode).
 */

#define ALIGNAS(X)	__attribute__((aligned(X)))

/**
 * @union Color
 * @brief Represents a 32-bit RGBA color with multiple access patterns.
 */
union Color {
	struct BgraColorType {
		uchar b; ///< Blue channel
		uchar g; ///< Green channel
		uchar r; ///< Red channel
		uchar a; ///< Alpha channel (or auxiliary data)
	} channels;
	uchar components[4]; ///< Array-style access
	uint bits; ///< Raw 32-bit integer access
};

/**
 * @brief Clamps an integer value to the [min, max] range and casts to uchar.
 */
uchar my_clamp(int val, int min, int max) {
	return (uchar)(val < min ? min : (val > max ? max : val));
}

/**
 * @brief Rescales an 8-bit color component to 5 bits with rounding.
 */
uchar round_to_5_bits(float val) {
	return my_clamp(val * 31.0f / 255.0f + 0.5f, 0, 31);
}

/**
 * @brief Rescales an 8-bit color component to 4 bits with rounding.
 */
uchar round_to_4_bits(float val) {
	return my_clamp(val * 15.0f / 255.0f + 0.5f, 0, 15);
}

/**
 * @brief ETC1 Luminance modulation tables.
 * Each row corresponds to a table index (0-7), providing 4 modulation values.
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

/// Map internal modulation indices to pixel index values.
__constant uchar g_mod_to_pix[4] = {3, 2, 0, 1};

/// Mapping of indices to sub-block texel numbers for different layout configurations.
__constant uchar g_idx_to_num[4][8] = {
	{0, 4, 1, 5, 2, 6, 3, 7},        
	{8, 12, 9, 13, 10, 14, 11, 15},  
	{0, 4, 8, 12, 1, 5, 9, 13},      
	{2, 6, 10, 14, 3, 7, 11, 15}     
};

/**
 * @brief Generates a new color by applying luminance modulation.
 * @param base The base color to modulate.
 * @param lum The luminance offset.
 * @return The modulated color, clamped to valid RGB range.
 */
union Color makeColor(union Color base, short lum) {
	int b = convert_int(base.channels.b) + lum;
	int g = convert_int(base.channels.g) + lum;
	int r = convert_int(base.channels.r) + lum;
	union Color color;
	color.channels.b = convert_uchar(my_clamp(b, 0, 255));
	color.channels.g = convert_uchar(my_clamp(g, 0, 255));
	color.channels.r = convert_uchar(my_clamp(r, 0, 255));
	return color;
}

/**
 * @brief Calculates the squared error between two colors.
 * Functional Utility: Used to evaluate compression quality during exhaustive search.
 */
uint getColorError(union Color u, union Color v) {
#ifdef USE_PERCEIVED_ERROR_METRIC
	float delta_b = convert_float(u.channels.b) - v.channels.b;
	float delta_g = convert_float(u.channels.g) - v.channels.g;
	float delta_r = convert_float(u.channels.r) - v.channels.r;
	return convert_uint(0.299f * delta_b * delta_b +
								 0.587f * delta_g * delta_g +
								 0.114f * delta_r * delta_r);
#else
	int delta_b = convert_int(u.channels.b) - v.channels.b;
	int delta_g = convert_int(u.channels.g) - v.channels.g;
	int delta_r = convert_int(u.channels.r) - v.channels.r;
	return delta_b * delta_b + delta_g * delta_g + delta_r * delta_r;
#endif
}

/**
 * @brief Packs two colors into a 444 (4-bit per channel) block format.
 */
void WriteColors444(uchar* block,
					union Color color0,
					union Color color1) {
	
	block[0] = (color0.channels.r & 0xf0) | (color1.channels.r >> 4);
	block[1] = (color0.channels.g & 0xf0) | (color1.channels.g >> 4);
	block[2] = (color0.channels.b & 0xf0) | (color1.channels.b >> 4);
}

/**
 * @brief Packs two colors into a 555 differential block format.
 * Algorithm: Computes 3-bit deltas between sub-blocks and encodes them using two's complement.
 */
void WriteColors555(uchar* block,
					union Color color0,
					union Color color1) {
	
	/// 2's complement translation table for 3-bit signed differences.
	uchar two_compl_trans_table[8] = {
		4, 5, 6, 7, 0, 1, 2, 3,
	};
	
	short delta_r =
	convert_short(color1.channels.r >> 3) - (color0.channels.r >> 3);
	
	short delta_g =
	convert_short(color1.channels.g >> 3) - (color0.channels.g >> 3);
	
	short delta_b =
	convert_short(color1.channels.b >> 3) - (color0.channels.b >> 3);
	
	block[0] = (color0.channels.r & 0xf8) | two_compl_trans_table[delta_r + 4];
	block[1] = (color0.channels.g & 0xf8) | two_compl_trans_table[delta_g + 4];
	block[2] = (color0.channels.b & 0xf8) | two_compl_trans_table[delta_b + 4];
}

/**
 * @brief Writes the chosen modulation table index into the block.
 */
void WriteCodewordTable(uchar* block,
						uchar sub_block_id,
						uchar table) {
	
	uchar shift = (2 + (3 - sub_block_id * 3));
	block[3] &= ~(0x07 << shift);
	block[3] |= table << shift;
}

/**
 * @brief Writes pixel-level modulation data into the block payload.
 */
void WritePixelData(uchar* block, uint pixel_data) {
	block[4] |= pixel_data >> 24;
	block[5] |= (pixel_data >> 16) & 0xff;
	block[6] |= (pixel_data >> 8) & 0xff;
	block[7] |= pixel_data & 0xff;
}

/**
 * @brief Sets the 'flip' bit in the block, indicating horizontal vs vertical split.
 */
void WriteFlip(uchar* block, bool flip) {
	block[3] &= ~0x01;
	block[3] |= convert_uchar(flip);
}

/**
 * @brief Sets the 'diff' bit in the block, indicating differential vs individual mode.
 */
void WriteDiff(uchar* block, bool diff) {
	block[3] &= ~0x02;
	block[3] |= convert_uchar(diff) << 1;
}

/**
 * @brief Extracts a 4x4 block from a larger source image.
 * Memory access: Optimized for row-major image layout.
 */
void ExtractBlock(uchar* dst, uchar* src, int width) {
	for (int j = 0; j < 4; ++j) {
		for (int k = 0; k < 16; k++) {
			dst[j * 16 + k] = *(src + k);
		}
		src += width * 4;
	}
}

/**
 * @brief Creates an internal Color representation from float BGR, rounded to 4 bits.
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
 * @brief Creates an internal Color representation from float BGR, rounded to 5 bits.
 */
union Color makeColor555(float* bgr) {
	uchar b5 = round_to_5_bits(bgr[0]);
	uchar g5 = round_to_5_bits(bgr[1]);
	uchar r5 = round_to_5_bits(bgr[2]);
	union Color bgr555;
	bgr555.channels.b = (b5 >> 2);
	bgr555.channels.g = (g5 >> 2);
	bgr555.channels.r = (r5 >> 2);
	bgr555.channels.a = 0x55;
	return bgr555;
}

/**
 * @brief Calculates the average color of 8 source pixels.
 */
void getAverageColor(union Color* src, float* avg_color)
{
	uint sum_b = 0, sum_g = 0, sum_r = 0;
	for (unsigned int i = 0; i < 8; ++i) {
		sum_b += src[i].channels.b;
		sum_g += src[i].channels.g;
		sum_r += src[i].channels.r;
	}
	float kInv8 = 1.0f / 8.0f;
	avg_color[0] = convert_float(sum_b) * kInv8;
	avg_color[1] = convert_float(sum_g) * kInv8;
	avg_color[2] = convert_float(sum_r) * kInv8;
}

/**
 * @brief Exhaustively finds the best luminance modulation for a sub-block.
 * Algorithm: Brute-force search over all 8 codeword tables and their 4 modifiers per pixel.
 * @return The minimum error found for this sub-block.
 */
unsigned long computeLuminance(uchar* block,
						   union Color* src,
						   union Color base,
						   int sub_block_id,
						   uchar* idx_to_num_tab,
						   unsigned long threshold)
{
	uint best_tbl_err = threshold;
	uchar best_tbl_idx = 0;
	uchar best_mod_idx[8][8];  

	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		union Color candidate_color[4];  
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			candidate_color[mod_idx] = makeColor(base, lum);
		}
		uint tbl_err = 0;
		for (unsigned int i = 0; i < 8; ++i) {
			uint best_mod_err = threshold;
			for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
				union Color color = candidate_color[mod_idx];
				uint mod_err = getColorError(src[i], color);
				if (mod_err < best_mod_err) {
					best_mod_idx[tbl_idx][i] = mod_idx;
					best_mod_err = mod_err;
					if (mod_err == 0) break;  
				}
			}
			tbl_err += best_mod_err;
			if (tbl_err > best_tbl_err) break;  
		}
		if (tbl_err < best_tbl_err) {
			best_tbl_err = tbl_err;
			best_tbl_idx = tbl_idx;
			if (tbl_err == 0) break;  
		}
	}
	WriteCodewordTable(block, sub_block_id, best_tbl_idx);
	uint pix_data = 0;
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
 * @brief Optimized compression path for blocks with identical pixels.
 * @return True if the block is solid and was successfully compressed.
 */
bool tryCompressSolidBlock(uchar* dst,
						   union Color* src,
						   unsigned long* error)
{
	for (unsigned int i = 1; i < 16; ++i) {
		if (src[i].bits != src[0].bits)
			return false;
	}
	for (int i = 0; i < 8; i++) {
		dst[i] = 0;
	}
	float src_color_float[3] = {convert_float(src->channels.b),
		convert_float(src->channels.g),
		convert_float(src->channels.r)};
	union Color base = makeColor555(src_color_float);
	WriteDiff(dst, true);
	WriteFlip(dst, false);
	WriteColors555(dst, base, base);
	uchar best_tbl_idx = 0;
	uchar best_mod_idx = 0;
	uint best_mod_err = UINT_MAX; 
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			union Color color = makeColor(base, lum);
			uint mod_err = getColorError(*src, color);
			if (mod_err < best_mod_err) {
				best_tbl_idx = tbl_idx;
				best_mod_idx = mod_idx;
				best_mod_err = mod_err;
				if (mod_err == 0) break;  
			}
		}
		if (best_mod_err == 0) break;
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
 * @brief Structural orchestration for encoding 4x4 blocks.
 * Partitioning: Decides between horizontal and vertical sub-block splits based on error heuristics.
 */
unsigned long compressBlock(uchar* dst,
							union Color* ver_src,
							union Color* hor_src,
							unsigned long threshold)
{
	unsigned long solid_error = 0;
	if (tryCompressSolidBlock(dst, ver_src, &solid_error)) {
		return solid_error;
	}
	union Color* sub_block_src[4] = {ver_src, ver_src + 8, hor_src, hor_src + 8};
	union Color sub_block_avg[4];
	bool use_differential[2] = {true, true};
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
	bool flip = sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1];
	for (int i = 0; i < 8; i++) {
		dst[i] = 0;
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
	unsigned long lumi_error1 = computeLuminance(dst, sub_block_src[sub_block_off_0],
								   sub_block_avg[sub_block_off_0], 0,
								   g_idx_to_num[sub_block_off_0],
								   threshold);
	unsigned long lumi_error2 = computeLuminance(dst, sub_block_src[sub_block_off_1],
								   sub_block_avg[sub_block_off_1], 1,
								   g_idx_to_num[sub_block_off_1],
								   threshold);
	return lumi_error1 + lumi_error2;
}


__kernel void kernel_test(__global uchar* src,
						__global uchar* dst)
{
	uint gid = get_global_id(0);
	uint lid = get_local_id(0);

	
}


#include "helper.hpp"

using namespace std;

// ... (Host-side OpenCL management code below, similar to previous files)
