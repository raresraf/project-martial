/**
 * @file kernel.cl
 * @brief OpenCL kernel and utility functions for ETC1 texture compression.
 *
 * This module implements the Ericsson Texture Compression (ETC1) algorithm.
 * It provides functions for color quantization, luminance table selection, 
 * and block-based compression of BGRA8888 images into ETC1 format.
 *
 * Domain: Graphics, HPC, Image Compression.
 */

/**
 * @union Color
 * @brief Represents a color in BGRA format with various access methods.
 */
union infoRGB
{
	struct BgraColorType
	{
		uchar b;
		uchar g;
		uchar r;
		uchar a;
	} channels;
	uchar components[4];
	uint bits;
};


typedef union infoRGB Color; 

/**
 * round_to_5_bits - Quantizes a color component to 5 bits.
 */
uchar round_to_5_bits(float val)
{
	return clamp(val * 31.0f / 255.0f + 0.5f, 0.0f, 31.0f);
}

/**
 * round_to_4_bits - Quantizes a color component to 4 bits.
 */
uchar round_to_4_bits(float val)
{
	return clamp(val * 15.0f / 255.0f + 0.5f, 0.0f, 15.0f);
}

/**
 * copy - Local implementation of memory copy.
 */
void copy(void *dst, void *src, int n)
{
	char *charSrc = (char *)src;
	char *charDst = (char *)dst;
 	int i;
	
	for (i = 0;i < n;i++)
	{
 	   charDst[i] = charSrc[i];
	}
}

/**
 * populate - Local implementation of memory set.
 */
void populate(void *pointer, int info, int n)
{
	int i = 0;
	uchar *unit = pointer;
  
	while(n > 0)
	{
    	*unit = info;
    	unit++;
    	n--;
    }
}

/**
 * Codeword tables used by ETC1 to store luminance modifiers.
 */
__constant short g_codeword_tables[8][4] =
{
	{-8, -2, 2, 8},
	{-17, -5, 5, 17},
	{-29, -9, 9, 29},
	{-42, -13, 13, 42},
	{-60, -18, 18, 60},
	{-80, -24, 24, 80},
	{-106, -33, 33, 106},
	{-183, -47, 47, 183}
};

/**
 * Mapping from luminance modifier indices to pixel indices.
 */
__constant uchar g_mod_to_pix[4] = {3, 2, 0, 1};

/**
 * Mapping for sub-block texel indices in different partitioning modes.
 */
__constant uchar g_idx_to_num[4][8] =
{
	{0, 4, 1, 5, 2, 6, 3, 7},        
	{8, 12, 9, 13, 10, 14, 11, 15},  
	{0, 4, 8, 12, 1, 5, 9, 13},      
	{2, 6, 10, 14, 3, 7, 11, 15}     
};

/**
 * makeColor - Applies a luminance modifier to a base color.
 */
Color makeColor(const Color *base, short lum)
{
	Color color;

	int b = (int)(base->channels.b) + lum;
	int g = (int)(base->channels.g) + lum;
	int r = (int)(base->channels.r) + lum;
	
	color.channels.b = (uchar)(clamp(b, 0, 255));
	color.channels.g = (uchar)(clamp(g, 0, 255));
	color.channels.r = (uchar)(clamp(r, 0, 255));
	
	return color;
}

/**
 * getColorError - Computes the squared error between two colors.
 */
uint getColorError(const Color *u, const Color *v)
{
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
 * WriteColors444 - Encodes two RGB444 colors into the compressed block.
 */
void WriteColors444(uchar *block, const Color *color0, const Color *color1)
{	
	block[0] = (color0->channels.r & 0xf0) | (color1->channels.r >> 4);
	block[1] = (color0->channels.g & 0xf0) | (color1->channels.g >> 4);
	block[2] = (color0->channels.b & 0xf0) | (color1->channels.b >> 4);
}

/**
 * WriteColors555 - Encodes an RGB555 color and its 3-bit delta into the block.
 */
void WriteColors555(uchar *block, const Color *color0, const Color *color1)
{
	const uchar two_compl_trans_table[8] =
	{
		4, 5, 6, 7, 0, 1, 2, 3
	};
	
	short delta_r = (short)(color1->channels.r >> 3) - (color0->channels.r >> 3);
	short delta_g = (short)(color1->channels.g >> 3) - (color0->channels.g >> 3);
	short delta_b = (short)(color1->channels.b >> 3) - (color0->channels.b >> 3);
	
	block[0] = (color0->channels.r & 0xf8) | two_compl_trans_table[delta_r + 4];
	block[1] = (color0->channels.g & 0xf8) | two_compl_trans_table[delta_g + 4];
	block[2] = (color0->channels.b & 0xf8) | two_compl_trans_table[delta_b + 4];
}

/**
 * WriteCodewordTable - Sets the codeword table index for a sub-block.
 */
void WriteCodewordTable(uchar *block, uchar sub_block_id, uchar table)
{
	uchar shift = (2 + (3 - sub_block_id * 3));
	block[3] &= ~(0x07 << shift);
	block[3] |= table << shift;
}

/**
 * WritePixelData - Encodes the pixel indices into the compressed block.
 */
void WritePixelData(uchar *block, uint pixel_data)
{
	block[4] |= pixel_data >> 24;
	block[5] |= (pixel_data >> 16) & 0xff;
	block[6] |= (pixel_data >> 8) & 0xff;
	block[7] |= pixel_data & 0xff;
}

/**
 * WriteFlip - Sets the partitioning mode (flip bit) of the block.
 */
void WriteFlip(uchar *block, bool flip)
{
	block[3] &= ~0x01;
	block[3] |= (uchar)(flip);
}

/**
 * WriteDiff - Sets the color encoding mode (differential bit) of the block.
 */
void WriteDiff(uchar *block, bool diff)
{
	block[3] &= ~0x02;
	block[3] |= (uchar)(diff) << 1;
}

/**
 * ExtractBlock - Copies a 4x4 pixel block from the source image.
 */
void ExtractBlock(uchar *dst, const uchar *src, int width)
{
	int j;

	for (j = 0; j < 4; j++)
	{
		copy((void *)(&dst[j * 4 * 4]), (void *)src, 4 * 4);
		src += width * 4;
	}
}

/**
 * makeColor444 - Creates an RGB444 color from floating point components.
 */
Color makeColor444(const float *bgr)
{
	Color bgr444;

	uchar b4 = round_to_4_bits(bgr[0]);
	uchar g4 = round_to_4_bits(bgr[1]);
	uchar r4 = round_to_4_bits(bgr[2]);

	bgr444.channels.b = (b4 << 4) | b4;
	bgr444.channels.g = (g4 << 4) | g4;
	bgr444.channels.r = (r4 << 4) | r4;
	
	bgr444.channels.a = 0x44;

	return bgr444;
}

/**
 * makeColor555 - Creates an RGB555 color from floating point components.
 */
Color makeColor555(const float *bgr)
{
	Color bgr555;

	uchar b5 = round_to_5_bits(bgr[0]);
	uchar g5 = round_to_5_bits(bgr[1]);
	uchar r5 = round_to_5_bits(bgr[2]);
	
	bgr555.channels.b = (b5 > 2);
	bgr555.channels.g = (g5 > 2);
	bgr555.channels.r = (r5 > 2);
	
	bgr555.channels.a = 0x55;

	return bgr555;
}

/**
 * getAverageColor - Calculates the average color for a set of 8 pixels.
 */
void getAverageColor(const Color *src, float *avg_color)
{
	uint sum_b = 0, sum_g = 0, sum_r = 0;
	const float kInv8 = 1.0f / 8.0f;
	
	for (uint i = 0; i < 8; i++)
	{
		sum_b += src[i].channels.b;
		sum_g += src[i].channels.g;
		sum_r += src[i].channels.r;
	}
	
	avg_color[0] = (float)(sum_b) * kInv8;
	avg_color[1] = (float)(sum_g) * kInv8;
	avg_color[2] = (float)(sum_r) * kInv8;
}

/**
 * computeLuminance - Searches for the best luminance modifier for a sub-block.
 */
ulong computeLuminance(uchar *block, const Color *src, const Color *base, int sub_block_id, const uchar index, ulong threshold)
{
	uint best_tbl_err = threshold;
	uchar best_tbl_idx = 0;
	uchar best_mod_idx[8][8];  

	/**
	 * Iterates over all codeword tables to find the one with the minimal error.
	 */
	for (uint tbl_idx = 0; tbl_idx < 8; tbl_idx++)
	{
		Color candidate_color[4];  
		
		for (uint mod_idx = 0; mod_idx < 4; mod_idx++)
		{
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			candidate_color[mod_idx] = makeColor(base, lum);
		}
		
		uint tbl_err = 0;
		
		for (uint i = 0; i < 8; i++)
		{
			uint best_mod_err = threshold;

			for (uint mod_idx = 0; mod_idx < 4; mod_idx++)
			{
				const Color color = candidate_color[mod_idx];
				uint mod_err = getColorError(&src[i], &color);
				
				if (mod_err < best_mod_err)
				{
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
		
		if (tbl_err < best_tbl_err)
		{
			best_tbl_err = tbl_err;
			best_tbl_idx = tbl_idx;
			
			if (tbl_err == 0)
				break;  
		}
	}

	WriteCodewordTable(block, sub_block_id, best_tbl_idx);

	uint pix_data = 0;

	/**
	 * Encodes the best modifier indices into the pixel data field.
	 */
	for (uint i = 0; i < 8; i++)
	{
		uchar mod_idx = best_mod_idx[best_tbl_idx][i];
		uchar pix_idx = g_mod_to_pix[mod_idx];
		
		uint lsb = pix_idx & 0x1;
		uint msb = pix_idx >> 1;
		
		int texel_num = g_idx_to_num[index][i];
		pix_data |= msb << (texel_num + 16);
		pix_data |= lsb << (texel_num);
	}

	WritePixelData(block, pix_data);

	return best_tbl_err;
}

/**
 * tryCompressSolidBlock - Attempts to compress a block with uniform color.
 */
bool tryCompressSolidBlock(uchar *dst, const Color *src, ulong *error)
{
	for (uint i = 1; i < 16; i++)
	{
		if (src[i].bits != src[0].bits)
			return false;
	}
	
	populate(dst, 0, 8);

	float src_color_float[3] = {(float)(src->channels.b), (float)(src->channels.g), (float)(src->channels.r)};
	Color base = makeColor555(src_color_float);
	
	WriteDiff(dst, true);
	WriteFlip(dst, false);
	WriteColors555(dst, &base, &base);
	
	uchar best_tbl_idx = 0;
	uchar best_mod_idx = 0;
	uint best_mod_err = UINT_MAX; 
	
	for (uint tbl_idx = 0; tbl_idx < 8; tbl_idx++)
	{
		for (uint mod_idx = 0; mod_idx < 4; mod_idx++)
		{
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			const Color color = makeColor(&base, lum);
			
			uint mod_err = getColorError(src, &color);
			
			if (mod_err < best_mod_err)
			{
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

	for (uint i = 0; i < 2; i++)
	{
		for (uint j = 0; j < 8; j++)
		{
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
 * compressBlock - Compresses a single 4x4 pixel block into ETC1 format.
 */
ulong compressBlock(uchar *dst, const Color *ver_src, const Color *hor_src, ulong threshold)
{
	ulong solid_error = 0;
	
	if (tryCompressSolidBlock(dst, ver_src, &solid_error))
	{
		return solid_error;
	}
	
	const Color *sub_block_src[4] = {ver_src, ver_src + 8, hor_src, hor_src + 8};
	
	Color sub_block_avg[4];
	bool use_differential[2] = {true, true};
	
	/**
	 * Determines the color encoding mode for each sub-block candidate.
	 */
	for (uint i = 0, j = 1; i < 4; i += 2, j += 2)
	{
		float avg_color_0[3];
		getAverageColor(sub_block_src[i], avg_color_0);
		Color avg_color_555_0 = makeColor555(avg_color_0);
		
		float avg_color_1[3];
		getAverageColor(sub_block_src[j], avg_color_1);
		Color avg_color_555_1 = makeColor555(avg_color_1);
		
		for (uint light_idx = 0; light_idx < 3; light_idx++)
		{
			int u = avg_color_555_0.components[light_idx] >> 3;
			int v = avg_color_555_1.components[light_idx] >> 3;
			
			int component_diff = v - u;

			if (component_diff < -4 || component_diff > 3)
			{
				use_differential[i / 2] = false;
				sub_block_avg[i] = makeColor444(avg_color_0);
				sub_block_avg[j] = makeColor444(avg_color_1);
			}
			else
			{
				sub_block_avg[i] = avg_color_555_0;
				sub_block_avg[j] = avg_color_555_1;
			}
		}
	}
	
	uint sub_block_err[4] = {0};

	for (uint i = 0; i < 4; i++)
	{
		for (uint j = 0; j < 8; j++)
		{
			sub_block_err[i] += getColorError(&sub_block_avg[i], &sub_block_src[i][j]);
		}
	}
	
	/**
	 * Selects the partitioning mode (vertical vs horizontal) based on minimum error.
	 */
	bool flip = sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1];
	
	populate(dst, 0, 8);
	
	WriteDiff(dst, use_differential[!!flip]);
	WriteFlip(dst, flip);
	
	uchar sub_block_off_0 = flip ? 2 : 0;
	uchar sub_block_off_1 = sub_block_off_0 + 1;
	
	if (use_differential[!!flip])
	{
		WriteColors555(dst, &sub_block_avg[sub_block_off_0], &sub_block_avg[sub_block_off_1]);
	}
	else
	{
		WriteColors444(dst, &sub_block_avg[sub_block_off_0], &sub_block_avg[sub_block_off_1]);
	}
	
	ulong lumi_error1 = 0, lumi_error2 = 0;
	
	/**
	 * Computes optimal luminance modifiers for both sub-blocks.
	 */
	lumi_error1 = computeLuminance(dst, sub_block_src[sub_block_off_0], &sub_block_avg[sub_block_off_0], 0, sub_block_off_0, threshold);

	lumi_error2 = computeLuminance(dst, sub_block_src[sub_block_off_1], &sub_block_avg[sub_block_off_1], 1, sub_block_off_1, threshold);
	
	return lumi_error1 + lumi_error2;
}

/**
 * @kernel execute
 * @brief Main entry point for parallel ETC1 compression on the GPU.
 */
__kernel void execute(__global uchar *src, __global uchar *dst, __global int *dims)
{
	int width = dims[0];
	int height = dims[1];
	
	Color ver_blocks[16];
	Color hor_blocks[16];
	
	int ycoord = get_global_id(0);
	int xcoord = get_global_id(1);

	int soffset = width * 16 * ycoord + xcoord * 16;
	int doffset = width * 2 * ycoord + xcoord * 8;

	const Color* row0 = src + soffset;
	const Color* row1 = row0 + width;
	const Color* row2 = row1 + width;
	const Color* row3 = row2 + width;
			
	/**
	 * Aggregates texels for vertical sub-block partitioning candidates.
	 */
	copy((void *)ver_blocks, (void *)row0, 8);
	copy((void *)ver_blocks + 2, (void *)row1, 8);
	copy((void *)ver_blocks + 4, (void *)row2, 8);
	copy((void *)ver_blocks + 6, (void *)row3, 8);
	copy((void *)ver_blocks + 8, (void *)row0 + 2, 8);
	copy((void *)ver_blocks + 10, (void *)row1 + 2, 8);
	copy((void *)ver_blocks + 12, (void *)row2 + 2, 8);
	copy((void *)ver_blocks + 14, (void *)row3 + 2, 8);
			
	/**
	 * Aggregates texels for horizontal sub-block partitioning candidates.
	 */
	copy(hor_blocks, row0, 16);
	copy(hor_blocks + 4, row1, 16);
	copy(hor_blocks + 8, row2, 16);
	copy(hor_blocks + 12, row3, 16);

	uchar aux[8];
	
	compressBlock(aux, ver_blocks, hor_blocks, INT_MAX);

	/**
	 * Commits the compressed block data to global memory.
	 */
	for(int i = 0;i < 8;i++)
	{
		dst[doffset + i] = aux[i];
	}
}

// ... rest of the file (texture_compress_skl.cpp etc.) truncated for brevity.
