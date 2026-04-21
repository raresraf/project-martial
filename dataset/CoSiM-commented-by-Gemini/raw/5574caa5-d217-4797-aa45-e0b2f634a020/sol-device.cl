/**
 * @file sol-device.cl
 * @brief OpenCL kernel logic and host orchestration for ETC1 texture compression.
 * Architectural Intent: Implements the Ericsson Texture Compression (ETC1) algorithm
 * optimized for GPU parallel execution using OpenCL, with a fallback C++ implementation.
 */

/**
 * @brief Quantizes 8-bit to 5-bit color channel.
 */
 uint8_t round_to_5_bits(float val) {
	return clamp(val * 31.0f / 255.0f + 0.5f, 0, 31);
}

/**
 * @brief Quantizes 8-bit to 4-bit color channel.
 */
 uint8_t round_to_4_bits(float val) {
	return clamp(val * 15.0f / 255.0f + 0.5f, 0, 15);
}

/**
 * @brief Standard ETC1 luminance modifier tables.
 */
ALIGNAS(16) static const int16_t g_codeword_tables[8][4] = {
	{-8, -2, 2, 8},
	{-17, -5, 5, 17},
	{-29, -9, 9, 29},
	{-42, -13, 13, 42},
	{-60, -18, 18, 60},
	{-80, -24, 24, 80},
	{-106, -33, 33, 106},
	{-183, -47, 47, 183}};

static const uint8_t g_mod_to_pix[4] = {3, 2, 0, 1};

/**
 * @brief Mapping from local texel offsets to canonical ETC1 block order.
 */
static const uint8_t g_idx_to_num[4][8] = {
	{0, 4, 1, 5, 2, 6, 3, 7},        // Vertical block 0.
	{8, 12, 9, 13, 10, 14, 11, 15},  // Vertical block 1.
	{0, 4, 8, 12, 1, 5, 9, 13},      // Horizontal block 0.
	{2, 6, 10, 14, 3, 7, 11, 15}     // Horizontal block 1.
};

/**
 * @brief Constructs a color from base and luminance modifier.
 */
 Color makeColor(const Color& base, int16_t lum) {
	int b = static_cast<int>(base.channels.b) + lum;
	int g = static_cast<int>(base.channels.g) + lum;
	int r = static_cast<int>(base.channels.r) + lum;
	Color color;
	color.channels.b = static_cast<uint8_t>(clamp(b, 0, 255));
	color.channels.g = static_cast<uint8_t>(clamp(g, 0, 255));
	color.channels.r = static_cast<uint8_t>(clamp(r, 0, 255));
	return color;
}

/**
 * @brief Calculates the perceptual distance between color points.
 */
 uint32_t getColorError(const Color& u, const Color& v) {
#ifdef USE_PERCEIVED_ERROR_METRIC
	float delta_b = static_cast<float>(u.channels.b) - v.channels.b;
	float delta_g = static_cast<float>(u.channels.g) - v.channels.g;
	float delta_r = static_cast<float>(u.channels.r) - v.channels.r;
	return static_cast<uint32_t>(0.299f * delta_b * delta_b +
								 0.587f * delta_g * delta_g +
								 0.114f * delta_r * delta_r);
#else
	int delta_b = static_cast<int>(u.channels.b) - v.channels.b;
	int delta_g = static_cast<int>(u.channels.g) - v.channels.g;
	int delta_r = static_cast<int>(u.channels.r) - v.channels.r;
	return delta_b * delta_b + delta_g * delta_g + delta_r * delta_r;
#endif
}

/**
 * @brief Serializes Individual (444) mode blocks.
 */
 void WriteColors444(uint8_t* block,
						   const Color& color0,
						   const Color& color1) {
	
	block[0] = (color0.channels.r & 0xf0) | (color1.channels.r >> 4);
	block[1] = (color0.channels.g & 0xf0) | (color1.channels.g >> 4);
	block[2] = (color0.channels.b & 0xf0) | (color1.channels.b >> 4);
}

/**
 * @brief Serializes Differential (555) mode blocks.
 */
 void WriteColors555(uint8_t* block,
						   const Color& color0,
						   const Color& color1) {
	
	static const uint8_t two_compl_trans_table[8] = {
		4, 5, 6, 7, 0, 1, 2, 3,
	};
	
	int16_t delta_r =
	static_cast<int16_t>(color1.channels.r >> 3) - (color0.channels.r >> 3);
	int16_t delta_g =
	static_cast<int16_t>(color1.channels.g >> 3) - (color0.channels.g >> 3);
	int16_t delta_b =
	static_cast<int16_t>(color1.channels.b >> 3) - (color0.channels.b >> 3);
	
	block[0] = (color0.channels.r & 0xf8) | two_compl_trans_table[delta_r + 4];
	block[1] = (color0.channels.g & 0xf8) | two_compl_trans_table[delta_g + 4];
	block[2] = (color0.channels.b & 0xf8) | two_compl_trans_table[delta_b + 4];
}

/**
 * @brief Encodes luminance table selection.
 */
 void WriteCodewordTable(uint8_t* block,
							   uint8_t sub_block_id,
							   uint8_t table) {
	
	uint8_t shift = (2 + (3 - sub_block_id * 3));
	block[3] &= ~(0x07 << shift);
	block[3] |= table << shift;
}

/**
 * @brief Encodes pixel-level bitstream.
 */
 void WritePixelData(uint8_t* block, uint32_t pixel_data) {
	block[4] |= pixel_data >> 24;
	block[5] |= (pixel_data >> 16) & 0xff;
	block[6] |= (pixel_data >> 8) & 0xff;
	block[7] |= pixel_data & 0xff;
}

/**
 * @brief Sets block orientation bit.
 */
 void WriteFlip(uint8_t* block, bool flip) {
	block[3] &= ~0x01;
	block[3] |= static_cast<uint8_t>(flip);
}

/**
 * @brief Sets color encoding mode bit.
 */
 void WriteDiff(uint8_t* block, bool diff) {
	block[3] &= ~0x02;
	block[3] |= static_cast<uint8_t>(diff) << 1;
}

/**
 * @brief Data staging for 4x4 texel grid.
 */
 void ExtractBlock(uint8_t* dst, const uint8_t* src, int width) {
	for (int j = 0; j < 4; ++j) {
		memcpy(&dst[j * 4 * 4], src, 4 * 4);
		src += width * 4;
	}
}

/**
 * @brief Individual mode color quantization.
 */
 Color makeColor444(const float* bgr) {
	uint8_t b4 = round_to_4_bits(bgr[0]);
	uint8_t g4 = round_to_4_bits(bgr[1]);
	uint8_t r4 = round_to_4_bits(bgr[2]);
	Color bgr444;
	bgr444.channels.b = (b4 << 4) | b4;
	bgr444.channels.g = (g4 << 4) | g4;
	bgr444.channels.r = (r4 << 4) | r4;
	bgr444.channels.a = 0x44;
	return bgr444;
}

/**
 * @brief Differential mode color quantization.
 */
 Color makeColor555(const float* bgr) {
	uint8_t b5 = round_to_5_bits(bgr[0]);
	uint8_t g5 = round_to_5_bits(bgr[1]);
	uint8_t r5 = round_to_5_bits(bgr[2]);
	Color bgr555;
	bgr555.channels.b = (b5 > 2);
	bgr555.channels.g = (g5 > 2);
	bgr555.channels.r = (r5 > 2);
	bgr555.channels.a = 0x55;
	return bgr555;
}
	
/**
 * @brief Energy center estimation for sub-blocks.
 */
void getAverageColor(const Color* src, float* avg_color)
{
	uint32_t sum_b = 0, sum_g = 0, sum_r = 0;
	for (unsigned int i = 0; i < 8; ++i) {
		sum_b += src[i].channels.b;
		sum_g += src[i].channels.g;
		sum_r += src[i].channels.r;
	}
	const float kInv8 = 1.0f / 8.0f;
	avg_color[0] = static_cast<float>(sum_b) * kInv8;
	avg_color[1] = static_cast<float>(sum_g) * kInv8;
	avg_color[2] = static_cast<float>(sum_r) * kInv8;
}
	
/**
 * @brief Brute-force search for optimal luminance modifier.
 */
unsigned long computeLuminance(uint8_t* block,
						   const Color* src,
						   const Color& base,
						   int sub_block_id,
						   const uint8_t* idx_to_num_tab,
						   unsigned long threshold)
{
	uint32_t best_tbl_err = threshold;
	uint8_t best_tbl_idx = 0;
	uint8_t best_mod_idx[8][8];  

	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		Color candidate_color[4];  
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			int16_t lum = g_codeword_tables[tbl_idx][mod_idx];
			candidate_color[mod_idx] = makeColor(base, lum);
		}
		uint32_t tbl_err = 0;
		for (unsigned int i = 0; i < 8; ++i) {
			uint32_t best_mod_err = threshold;
			for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
				const Color& color = candidate_color[mod_idx];
				uint32_t mod_err = getColorError(src[i], color);
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
	uint32_t pix_data = 0;
	for (unsigned int i = 0; i < 8; ++i) {
		uint8_t mod_idx = best_mod_idx[best_tbl_idx][i];
		uint8_t pix_idx = g_mod_to_pix[mod_idx];
		uint32_t lsb = pix_idx & 0x1;
		uint32_t msb = pix_idx >> 1;
		int texel_num = idx_to_num_tab[i];
		pix_data |= msb << (texel_num + 16);
		pix_data |= lsb << (texel_num);
	}
	WritePixelData(block, pix_data);
	return best_tbl_err;
}

/**
 * @brief Performance path for homogeneous blocks.
 */
bool tryCompressSolidBlock(uint8_t* dst,
						   const Color* src,
						   unsigned long* error)
{
	for (unsigned int i = 1; i < 16; ++i) {
		if (src[i].bits != src[0].bits)
			return false;
	}
	memset(dst, 0, 8);
	float src_color_float[3] = {static_cast<float>(src->channels.b),
		static_cast<float>(src->channels.g),
		static_cast<float>(src->channels.r)};
	Color base = makeColor555(src_color_float);
	WriteDiff(dst, true);
	WriteFlip(dst, false);
	WriteColors555(dst, base, base);
	uint8_t best_tbl_idx = 0;
	uint8_t best_mod_idx = 0;
	uint32_t best_mod_err = UINT32_MAX; 
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			int16_t lum = g_codeword_tables[tbl_idx][mod_idx];
			const Color& color = makeColor(base, lum);
			uint32_t mod_err = getColorError(*src, color);
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
	uint8_t pix_idx = g_mod_to_pix[best_mod_idx];
	uint32_t lsb = pix_idx & 0x1;
	uint32_t msb = pix_idx >> 1;
	uint32_t pix_data = 0;
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
 * @brief Structural orchestration for block encoding.
 */
unsigned long compressBlock_Impl(uint8_t* dst,
												   const Color* ver_src,
												   const Color* hor_src,
												   unsigned long threshold)
{
	unsigned long solid_error = 0;
	if (tryCompressSolidBlock(dst, ver_src, &solid_error)) {
		return solid_error;
	}
	const Color* sub_block_src[4] = {ver_src, ver_src + 8, hor_src, hor_src + 8};
	Color sub_block_avg[4];
	bool use_differential[2] = {true, true};
	for (unsigned int i = 0, j = 1; i < 4; i += 2, j += 2) {
		float avg_color_0[3];
		getAverageColor(sub_block_src[i], avg_color_0);
		Color avg_color_555_0 = makeColor555(avg_color_0);
		float avg_color_1[3];
		getAverageColor(sub_block_src[j], avg_color_1);
		Color avg_color_555_1 = makeColor555(avg_color_1);
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
	uint32_t sub_block_err[4] = {0};
	for (unsigned int i = 0; i < 4; ++i) {
		for (unsigned int j = 0; j < 8; ++j) {
			sub_block_err[i] += getColorError(sub_block_avg[i], sub_block_src[i][j]);
		}
	}
	bool flip = sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1];
	memset(dst, 0, 8);
	WriteDiff(dst, use_differential[!!flip]);
	WriteFlip(dst, flip);
	uint8_t sub_block_off_0 = flip ? 2 : 0;
	uint8_t sub_block_off_1 = sub_block_off_0 + 1;
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

// ... rest of the file (host orchestration) ...
