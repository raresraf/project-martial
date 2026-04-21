/**
 * @file sol_device.cl
 * @brief OpenCL kernel and utility functions for ETC1 texture compression.
 *
 * This module implements the Ericsson Texture Compression (ETC1) algorithm.
 * It provides functions for color quantization, luminance table selection, 
 * and block-based compression of BGRA8888 images into ETC1 format.
 *
 * Domain: Graphics, HPC, Image Compression.
 */

/**
 * my_clamp - Local implementation of a clamp utility.
 */
inline uchar my_clamp(uchar val, uchar min, uchar max) {
	return val < min ? min : (val > max ? max : val);
}

/**
 * round_to_5_bits - Quantizes a color component to 5 bits.
 */
inline uchar round_to_5_bits(float val) {
	return my_clamp(val * 31.0f / 255.0f + 0.5f, 0, 31);
}

/**
 * round_to_4_bits - Quantizes a color component to 4 bits.
 */
inline uchar round_to_4_bits(float val) {
	return my_clamp(val * 15.0f / 255.0f + 0.5f, 0, 15);
}

/**
 * @union Color
 * @brief Represents a color in BGRA format with various access methods.
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
 * Codeword tables used by ETC1 to store luminance modifiers.
 */
__attribute__((aligned(16))) static constant short g_codeword_tables[8][4] = {
	{-8, -2, 2, 8},
	{-17, -5, 5, 17},
	{-29, -9, 9, 29},
	{-42, -13, 13, 42},
	{-60, -18, 18, 60},
	{-80, -24, 24, 80},
	{-106, -33, 33, 106},
	{-183, -47, 47, 183}};

/**
 * Mapping from luminance modifier indices to pixel indices.
 */
static constant uchar g_mod_to_pix[4] = {3, 2, 0, 1};

/**
 * Mapping for sub-block texel indices in different partitioning modes.
 */
static constant uchar g_idx_to_num[4][8] = {
	{0, 4, 1, 5, 2, 6, 3, 7},        
	{8, 12, 9, 13, 10, 14, 11, 15},  
	{0, 4, 8, 12, 1, 5, 9, 13},      
	{2, 6, 10, 14, 3, 7, 11, 15}     
};

/**
 * makeColor - Applies a luminance modifier to a base color.
 */
inline union Color makeColor( union Color* base, short lum) {
	int b = (int)(base->channels.b) + lum;
	int g = (int)(base->channels.g) + lum;
	int r = (int)(base->channels.r) + lum;
	union Color color;
	color.channels.b = (uchar)(clamp(b, 0, 255));
	color.channels.g = (uchar)(clamp(g, 0, 255));
	color.channels.r = (uchar)(clamp(r, 0, 255));
	return color;
}

/**
 * getColorError - Computes the squared error between two colors.
 */
inline uint getColorError( union Color* u,  union Color* v) {
	int delta_b = (int)(u->channels.b) - v->channels.b;
	int delta_g = (int)(u->channels.g) - v->channels.g;
	int delta_r = (int)(u->channels.r) - v->channels.r;
	return delta_b * delta_b + delta_g * delta_g + delta_r * delta_r;

}

/**
 * WriteColors444 - Encodes two RGB444 colors into the compressed block.
 */
inline void WriteColors444(uchar* block,
						    union Color* color0,
						    union Color* color1) {
	
	block[0] = (color0->channels.r & 0xf0) | (color1->channels.r >> 4);
	block[1] = (color0->channels.g & 0xf0) | (color1->channels.g >> 4);
	block[2] = (color0->channels.b & 0xf0) | (color1->channels.b >> 4);
}

/**
 * WriteColors555 - Encodes an RGB555 color and its 3-bit delta into the block.
 */
inline void WriteColors555(uchar* block,
						    union Color* color0,
						    union Color* color1) {
	
	const uchar two_compl_trans_table[8] = {
		4, 5, 6, 7, 0, 1, 2, 3, 
	};
	
	short delta_r =
	(short)(color1->channels.r >> 3) - (color0->channels.r >> 3);
	short delta_g =
	(short)(color1->channels.g >> 3) - (color0->channels.g >> 3);
	short delta_b =
	(short)(color1->channels.b >> 3) - (color0->channels.b >> 3);
	
	block[0] = (color0->channels.r & 0xf8) | two_compl_trans_table[delta_r + 4];
	block[1] = (color0->channels.g & 0xf8) | two_compl_trans_table[delta_g + 4];
	block[2] = (color0->channels.b & 0xf8) | two_compl_trans_table[delta_b + 4];
}

/**
 * WriteCodewordTable - Sets the codeword table index for a sub-block.
 */
inline void WriteCodewordTable(uchar* block,
							   uchar sub_block_id,
							   uchar table) {
	
	uchar shift = (2 + (3 - sub_block_id * 3));
	block[3] &= ~(0x07 << shift);
	block[3] |= table << shift;
}

/**
 * WritePixelData - Encodes the pixel indices into the compressed block.
 */
inline void WritePixelData(uchar* block, int pixel_data) {
	block[4] |= pixel_data >> 24;
	block[5] |= (pixel_data >> 16) & 0xff;
	block[6] |= (pixel_data >> 8) & 0xff;
	block[7] |= pixel_data & 0xff;
}

/**
 * WriteFlip - Sets the partitioning mode (flip bit) of the block.
 */
inline void WriteFlip(uchar* block, uchar flip) {
	block[3] &= ~0x01;
	block[3] |= flip;
}

/**
 * WriteDiff - Sets the color encoding mode (differential bit) of the block.
 */
inline void WriteDiff(uchar* block, uchar diff) {
	block[3] &= ~0x02;
	block[3] |= (diff) << 1;
}

/**
 * my_memcpy - Local implementation of memory copy.
 */
void my_memcpy(void *dest, void *src, uint n)
{
   char *csrc = (char *)src;
   char *cdest = (char *)dest;
 
   int i;
   for (i=0; i<n; i++)
       cdest[i] = csrc[i];
}

/**
 * ExtractBlock - Copies a 4x4 pixel block from the source image.
 */
inline void ExtractBlock(uchar* dst, uchar* src, int width) {
	int j;
	for (j = 0; j < 4; ++j) {
		my_memcpy(&dst[j * 4 * 4], (void *)src, 4 * 4);
		src += width * 4;
	}
}

/**
 * makeColor444 - Creates an RGB444 color from floating point components.
 */
inline union Color makeColor444(float* bgr) {
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
 * makeColor555 - Creates an RGB555 color from floating point components.
 */
inline union Color makeColor555(float* bgr) {
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
 * getAverageColor - Calculates the average color for a set of 8 pixels.
 */
void getAverageColor(union Color* src, float* avg_color)
{
	uint sum_b = 0, sum_g = 0, sum_r = 0;
	unsigned int i;


	for (i = 0; i < 8; ++i) {
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
 * computeLuminance - Searches for the best luminance modifier for a sub-block.
 */
unsigned long computeLuminance(uchar* block,
						    union Color* src,
						    union Color* base,
						   int sub_block_id,
						   constant uchar* idx_to_num_tab,
						   unsigned long threshold)
{
	uint best_tbl_err = threshold;
	uchar best_tbl_idx = 0;
	uchar best_mod_idx[8][8];  

	/**
	 * Iterates over all codeword tables to find the one with the minimal error.
	 */
	uint tbl_idx = 0;
	for (tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		union Color candidate_color[4];  
		
		uint mod_idx;
		for (mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			candidate_color[mod_idx] = makeColor(base, lum);
		}
		
		uint tbl_err = 0;
		uint i;
		for (i = 0; i < 8; ++i) {
			uint best_mod_err = threshold;
			uint mod_idx;
			for (mod_idx = 0; mod_idx < 4; ++mod_idx) {
				 union Color *color = &candidate_color[mod_idx];
				
				uint mod_err = getColorError(&src[i], color);
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
	uint i;
	/**
	 * Encodes the best modifier indices into the pixel data field.
	 */
	for (i = 0; i < 8; ++i) {
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
 * my_memset - Local implementation of memory set.
 */
void* my_memset(void* pointer, int c, int size) {
    if ( pointer != NULL && size > 0 ) {
        uchar* pChar =  pointer;
        int i = 0;
        for ( i = 0; i < size; ++i) {
            unsigned char temp = (unsigned char) c;
            *pChar++ = temp; 
        }
    }
    return pointer;
}

/**
 * tryCompressSolidBlock - Attempts to compress a block with uniform color.
 */
uchar tryCompressSolidBlock(uchar* dst,
						    union Color* src,
						   unsigned long* error)
{
	uint i;
	uchar my_true = 0x01;
	uchar my_false = 0x00;
	for (i = 1; i < 16; ++i) {
		if (src[i].bits != src[0].bits)
			return my_false;
	}
	
	my_memset(dst, 0, 8);
	
	float src_color_float[3] = {(float)(src->channels.b),
		(float)(src->channels.g),
		(float)(src->channels.r)};
	union Color base = makeColor555(src_color_float);
	
	WriteDiff(dst, my_true);
	WriteFlip(dst, my_false);
	WriteColors555(dst, &base, &base);
	
	uchar best_tbl_idx = 0;
	uchar best_mod_idx = 0;
	uint best_mod_err = UINT_MAX; 
	
	uint tbl_idx;
	for (tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		uint mod_idx;
		for (mod_idx = 0; mod_idx < 4; ++mod_idx) {
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
	for (i = 0; i < 2; ++i) {
		uint j;
		for (j = 0; j < 8; ++j) {
			int texel_num = g_idx_to_num[i][j];
			pix_data |= msb << (texel_num + 16);
			pix_data |= lsb << (texel_num);
		}
	}
	
	WritePixelData(dst, pix_data);
	*error = 16 * best_mod_err;
	return my_true;
}

/**
 * compressBlock - Compresses a single 4x4 pixel block into ETC1 format.
 */
unsigned long compressBlock(uchar* dst,  union Color* ver_src,
									union Color* hor_src,
									unsigned long threshold)
{
	unsigned long solid_error = 0;
	if (tryCompressSolidBlock(dst, ver_src, &solid_error)) {
		return solid_error;
	}
	
	union Color* sub_block_src[4] = {ver_src, ver_src + 8, hor_src, hor_src + 8};
	
	union Color sub_block_avg[4];
	uchar use_differential[2] = {0x01, 0x01};
	
	/**
	 * Determines the color encoding mode for each sub-block candidate.
	 */
	uint i, j;
	for (i = 0, j = 1; i < 4; i += 2, j += 2) {
		float avg_color_0[3];
		getAverageColor(sub_block_src[i], avg_color_0);
		union Color avg_color_555_0 = makeColor555(avg_color_0);
		
		float avg_color_1[3];
		getAverageColor(sub_block_src[j], avg_color_1);
		union Color avg_color_555_1 = makeColor555(avg_color_1);
		uint light_idx;
		for (light_idx = 0; light_idx < 3; ++light_idx) {
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
	for (i = 0; i < 4; ++i) {
		uint j;
		for (j = 0; j < 8; ++j) {
			sub_block_err[i] += getColorError(&sub_block_avg[i], &sub_block_src[i][j]);
		}
	}
	
	/**
	 * Selects the partitioning mode (vertical vs horizontal) based on minimum error.
	 */
	uchar flip;
	if (sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1])
		flip = 0x01;
	else
		flip = 0x00;
	
	my_memset(dst, 0, 8);
	
	WriteDiff(dst, use_differential[!!flip]);
	WriteFlip(dst, flip);
	uchar sub_block_off_0;
	if (flip == 0x01)
		sub_block_off_0 = 2;
	else
		sub_block_off_0 = 0;

	uchar sub_block_off_1 = sub_block_off_0 + 1;

	if (use_differential[!!flip]) {
		WriteColors555(dst, &sub_block_avg[sub_block_off_0],
					   &sub_block_avg[sub_block_off_1]);
	} else {
		WriteColors444(dst, &sub_block_avg[sub_block_off_0],
					   &sub_block_avg[sub_block_off_1]);
	}
	
	ulong lumi_error1 = 0, lumi_error2 = 0;
	
	/**
	 * Computes optimal luminance modifiers for both sub-blocks.
	 */
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
 * @kernel compress
 * @brief Main entry point for parallel ETC1 compression on the GPU.
 */
__kernel void
compress(__global uchar* matSRC,
		__global uchar* matDST,
		int height,
		int width)
{
	union Color ver_blocks[16];
	union Color hor_blocks[16];

	int y = get_global_id(0);
	int x = get_global_id(1);

	int i;
	int size = 8;
	uchar* cpy_dst;

	/**
	 * Aggregates texels for partitioning candidates.
	 */
	for (i = 0; i < 4; i++) {
		cpy_dst = (uchar *)(ver_blocks + i * 2);
		int j;
		for (j = 0; j < size; j++) {
			cpy_dst[j] = matSRC[y * width * 4 * 4 
				+ width * 4 * i + x * 4 * 4 + j];
		}
	}
	
	for (i = 0; i < 4; i++) {
		cpy_dst = (uchar *)(ver_blocks + (8 + i * 2));
		int j;
		for (j = 0; j < size; j++) {
			cpy_dst[j] = matSRC[y * width * 4 * 4 
				+ width * 4 * i + 2 * 4 + x * 4 * 4 + j];
		}
	}
	
	size = 16;
	for (i = 0; i < 4; i++) {
		cpy_dst = (uchar *)(hor_blocks + i * 4);
		int j;
		for (j = 0; j < size; j++) {
			cpy_dst[j] = matSRC[y * width * 4 * 4 
				+ width * 4 * i + x * 4 * 4 + j];
		}
	}

	uchar dst[8];
	compressBlock(dst, ver_blocks, hor_blocks, INT_MAX);

	/**
	 * Commits the compressed block data to global memory.
	 */
	for (i = 0; i < 8; i++)
		matDST[y * width * 2 + 8 *x + i] = dst[i];
}

// ... rest of the file (host code etc.) truncated for brevity.
