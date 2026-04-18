/**
 * @file gpu_kernel.cl
 * @brief An OpenCL kernel and C++ host code for a texture compression algorithm.
 *
 * This file contains both the OpenCL device code for compressing 4x4 pixel blocks
 * and the C++ host code responsible for setting up the OpenCL environment,
 * managing memory, and dispatching the compression kernel. The compression
 * algorithm appears to be a variant of a block-based texture compression format,
 * possibly similar to ETC, using differential and solid color modes and
 * luminance modulation.
 */

// OpenCL Device Code Section

typedef union UColor {
  struct BgraColorType {
    uchar b;
    uchar g;
    uchar r;
    uchar a;
  } channels;
  uchar components[4];
  uint bits;
} Color;

/**
 * @brief Codeword tables for luminance modulation.
 * Stored in __constant memory for fast, cached access by all work-items.
 */
__attribute__ ((aligned (16))) __constant short g_codeword_tables[8][4] = {
  {-8, -2, 2, 8},
	{-17, -5, 5, 17},
	{-29, -9, 9, 29},
	{-42, -13, 13, 42},
	{-60, -18, 18, 60},
	{-80, -24, 24, 80},
	{-106, -33, 33, 106},
	{-183, -47, 47, 183}
};

// Maps a 2-bit modifier index to a 2-bit pixel index.
__constant uchar g_mod_to_pix[4] = {3, 2, 0, 1};

// Tables for reordering pixel indices based on the block 'flip' mode.
__constant uchar g_idx_to_num[4][8] = {
	{0, 4, 1, 5, 2, 6, 3, 7},
	{8, 12, 9, 13, 10, 14, 11, 15},
	{0, 4, 8, 12, 1, 5, 9, 13},
	{2, 6, 10, 14, 3, 7, 11, 15}
};

// Clamps a signed integer to the given range.
int clamp_signed(int val, int min, int max) {
	return val < min ? min : (val > max ? max : val);
}

// Clamps an unsigned integer to the given range.
uint clamp_unsigned(uint val, uint min, uint max) {
	return val < min ? min : (val > max ? max : val);
}

// Quantizes a 0-255 float value to 5 bits.
uchar round_to_5_bits(float val) {
	return (uchar) clamp_unsigned((uint) (val * 31.0f / 255.0f + 0.5f), 0, 31);
}

// Quantizes a 0-255 float value to 4 bits.
uchar round_to_4_bits(float val) {
	return (uchar) clamp_unsigned((uint) (val * 15.0f / 255.0f + 0.5f), 0, 15);
}

// Creates a new color by applying a luminance offset to a base color.
Color makeColor(const Color *base, short lum) {
	int b = (int) (base->channels.b) + lum;
	int g = (int) (base->channels.g) + lum;
	int r = (int) (base->channels.r) + lum;
	Color color;
	color.channels.b = (uchar) clamp_signed(b, 0, 255);
	color.channels.g = (uchar) clamp_signed(g, 0, 255);
	color.channels.r = (uchar) clamp_signed(r, 0, 255);
	return color;
}

// Calculates the squared Euclidean distance between two colors.
uint getColorError(const Color *u, const Color *v) {
	int delta_b = (int) (u->channels.b) - v->channels.b;
	int delta_g = (int) (u->channels.g) - v->channels.g;
	int delta_r = (int) (u->channels.r) - v->channels.r;
	return delta_b * delta_b + delta_g * delta_g + delta_r * delta_r;
}

// Writes two 4-bit-per-channel colors to the compressed block data.
void WriteColors444(__global uchar *block,
						   const Color *color0,
						   const Color *color1) {
	
	block[0] = (color0->channels.r & 0xf0) | (color1->channels.r >> 4);
	block[1] = (color0->channels.g & 0xf0) | (color1->channels.g >> 4);
	block[2] = (color0->channels.b & 0xf0) | (color1->channels.b >> 4);
}

// Writes two 5-bit-per-channel colors to the compressed block data in differential mode.
void WriteColors555(__global uchar *block,
						   const Color *color0,
						   const Color *color1) {
	
	const uchar two_compl_trans_table[8] = {
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
  (short) (color1->channels.r >> 3) - (color0->channels.r >> 3);
	short delta_g =
	(short) (color1->channels.g >> 3) - (color0->channels.g >> 3);
	short delta_b =
	(short) (color1->channels.b >> 3) - (color0->channels.b >> 3);
	
	
	block[0] = (color0->channels.r & 0xf8) | two_compl_trans_table[delta_r + 4];
	block[1] = (color0->channels.g & 0xf8) | two_compl_trans_table[delta_g + 4];
	block[2] = (color0->channels.b & 0xf8) | two_compl_trans_table[delta_b + 4];
}

// Writes the codeword table index for a sub-block.
void WriteCodewordTable(__global uchar *block,
							   uchar sub_block_id,
							   uchar table) {

	uchar shift = (2 + (3 - sub_block_id * 3));
	block[3] &= ~(0x07 << shift);
	block[3] |= table << shift;
}

// Writes the 32-bit packed pixel indices to the compressed block.
void WritePixelData(__global uchar* block, uint pixel_data) {
	block[4] |= pixel_data >> 24;
	block[5] |= (pixel_data >> 16) & 0xff;
	block[6] |= (pixel_data >> 8) & 0xff;
	block[7] |= pixel_data & 0xff;
}

// Writes the 'flip' bit, which determines the sub-block partitioning (2x4 vs 4x2).
void WriteFlip(__global uchar *block, int flip) {
	block[3] &= ~0x01;
	block[3] |= (uchar) flip;
}

// Writes the 'diff' bit, which determines if differential color encoding is used.
void WriteDiff(__global uchar *block, int diff) {
	block[3] &= ~0x02;
	block[3] |= ((uchar) diff) << 1;
}

// Extracts a 4x4 block of pixels from the source image.
void ExtractBlock(__global uchar *dst, const uchar *src, int width) {
  int i, j;

	for (j = 0; j < 4; ++j) {
    for(i = 0; i < 16; i++)
		  dst[j * 4 * 4 + i] = src[i];
		src += width * 4;
	}
}

// Creates a 4-bit-per-channel color, expanding it back to 8-bit.
Color makeColor444(const float *bgr) {
	uchar b4 = round_to_4_bits(bgr[0]);
	uchar g4 = round_to_4_bits(bgr[1]);
	uchar r4 = round_to_4_bits(bgr[2]);
	Color bgr444;
	bgr444.channels.b = (b4 << 4) | b4;
	bgr444.channels.g = (g4 << 4) | g4;
	bgr444.channels.r = (r4 << 4) | r4;
	
	bgr444.channels.a = 0x44;
	return bgr444;
}

// Creates a 5-bit-per-channel color, expanding it back to 8-bit.
Color makeColor555(const float *bgr) {
	uchar b5 = round_to_5_bits(bgr[0]);
	uchar g5 = round_to_5_bits(bgr[1]);
	uchar r5 = round_to_5_bits(bgr[2]);
	Color bgr555;
	bgr555.channels.b = (b5 < 3); // Bug: should be '<<'
	bgr555.channels.g = (g5 < 3); // Bug: should be '<<'
	bgr555.channels.r = (r5 < 3); // Bug: should be '<<'
	
	bgr555.channels.a = 0x55;
	return bgr555;
}

// Computes the average color of an 8-pixel sub-block.
void getAverageColor(const Color *src, float *avg_color)
{
	uint sum_b = 0, sum_g = 0, sum_r = 0;
    int i;
	
	for (i = 0; i < 8; ++i) {
		sum_b += src[i].channels.b;
		sum_g += src[i].channels.g;
		sum_r += src[i].channels.r;
	}
	
	const float kInv8 = 1.0f / 8.0f;
	avg_color[0] = (float) sum_b * kInv8;
	avg_color[1] = (float) sum_g * kInv8;
	avg_color[2] = (float) sum_r * kInv8;
}

/**
 * @brief Finds the best luminance table and modifier indices for a sub-block.
 * @return The total error for the best configuration.
 *
 * This is the core of the compression logic. It performs a search over all
 * codeword tables and modifiers to find the combination that minimizes the
 * color error for the 8 pixels in the sub-block (Vector Quantization).
 */
unsigned long computeLuminance(__global uchar *block,
						   const Color *src,
						   const Color *base,
						   int sub_block_id,
						   __constant uchar *idx_to_num_tab,
						   unsigned long threshold)
{
	uint best_tbl_err = threshold;
	uchar best_tbl_idx = 0;
	uchar best_mod_idx[8][8];
    unsigned int tbl_idx, mod_idx, i;
	uint pix_data = 0;

	// Block Logic: Iterate through all 8 codeword tables.
	for (tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		
		// Pre-calculate the 4 candidate colors for the current table.
		Color candidate_color[4];
		for (mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			candidate_color[mod_idx] = makeColor(base, lum);
		}
		
		uint tbl_err = 0;
		
		// Block Logic: For each of the 8 pixels in the sub-block...
		for (i = 0; i < 8; ++i) {
			
			// ...find the best modifier that minimizes color error.
			uint best_mod_err = threshold;
			for (mod_idx = 0; mod_idx < 4; ++mod_idx) {
				const Color *color = &candidate_color[mod_idx];
				
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
		
		// If the current table gives a lower total error, update our best choice.
		if (tbl_err < best_tbl_err) {
			best_tbl_err = tbl_err;
			best_tbl_idx = tbl_idx;
			
			if (tbl_err == 0)
				break;
		}
	}

	// Write the chosen table index to the block data.
	WriteCodewordTable(block, sub_block_id, best_tbl_idx);

	// Pack the best modifier indices for all 8 pixels into a 32-bit integer.
	for (i = 0; i < 8; ++i) {
		uchar mod_idx = best_mod_idx[best_tbl_idx][i];
		uchar pix_idx = g_mod_to_pix[mod_idx];
		
		uint lsb = pix_idx & 0x1;
		uint msb = pix_idx >> 1;
		
		int texel_num = idx_to_num_tab[i];
		pix_data |= msb << (texel_num + 16);
		pix_data |= lsb << (texel_num);
	}

	// Write the packed pixel data.
	WritePixelData(block, pix_data);

	return best_tbl_err;
}

// Attempts to compress a 4x4 block as a single solid color.
int tryCompressSolidBlock(__global uchar *dst,
						   const Color *src,
						   unsigned long *error)
{
    unsigned int i, tbl_idx, mod_idx, j;

	// Pre-condition: Check if all 16 pixels are the same color.
	for (i = 1; i < 16; ++i) {
		if (src[i].bits != src[0].bits)
			return false;
	}

  for (i = 0; i < 8; i++)
    dst[i] = 0;
	
	float src_color_float[3] = {(float) src->channels.b,
		(float) src->channels.g,
		(float) src->channels.r};
	Color base = makeColor555(src_color_float);
	
	WriteDiff(dst, true);
	WriteFlip(dst, false);
	WriteColors555(dst, &base, &base);
	
	uchar best_tbl_idx = 0;
	uchar best_mod_idx = 0;
	uint best_mod_err = 0x7FFFFFFF;

	// Find the best luminance modifier for the solid color.
	for (tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		for (mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			const Color color = makeColor(&base, lum);
			
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
		for (j = 0; j < 8; ++j) {
			int texel_num = g_idx_to_num[i][j];
			pix_data |= msb << (texel_num + 16);
			pix_data |= lsb << (texel_num);
		}
	}
	
	WritePixelData(dst, pix_data);
	*error = 16 * best_mod_err;
	return true;
}

// Main compression function for a 4x4 block.
unsigned long compressBlock(__global uchar *dst,
												   const Color *ver_src,
												   const Color *hor_src,
												   unsigned long threshold)
{
	unsigned long solid_error = 0;
    unsigned int i, j, light_idx;
	// Fast path for solid color blocks.
	if (tryCompressSolidBlock(dst, ver_src, &solid_error)) {
		return solid_error;
	}
	
	const Color *sub_block_src[4] = {ver_src, ver_src + 8, hor_src, hor_src + 8};
	
	Color sub_block_avg[4];
	int use_differential[2] = {1, 1};
	
	// Determine if differential mode should be used for each pair of sub-blocks.
	for (i = 0, j = 1; i < 4; i += 2, j += 2) {
		float avg_color_0[3];
		getAverageColor(sub_block_src[i], avg_color_0);

		Color avg_color_555_0 = makeColor555(avg_color_0);
		
		float avg_color_1[3];
		getAverageColor(sub_block_src[j], avg_color_1);
		Color avg_color_555_1 = makeColor555(avg_color_1);
		
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
		for (j = 0; j < 8; ++j) {
			sub_block_err[i] += getColorError(&sub_block_avg[i], &sub_block_src[i][j]);
		}
	}
	
    // Determine the best flip mode by comparing errors.
    int flip =
	(sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1]) ? 1 : 0;
	
	
	for (i = 0; i < 8; i++)
    dst[i] = 0;
	
	WriteDiff(dst, use_differential[!!flip]);
	WriteFlip(dst, flip);
	
	uchar sub_block_off_0 = flip ? 2 : 0;
	uchar sub_block_off_1 = sub_block_off_0 + 1;
	
	if (use_differential[!!flip]) {
		WriteColors555(dst, &sub_block_avg[sub_block_off_0],
					   &sub_block_avg[sub_block_off_1]);
	} else {
		WriteColors444(dst, &sub_block_avg[sub_block_off_0],
					   &sub_block_avg[sub_block_off_1]);
	}
	
	unsigned long lumi_error1 = 0, lumi_error2 = 0;
	
	// Compress the two chosen sub-blocks.
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

// A simple memcpy implementation for the device.
void my_memcpy(uchar *dst, __global uchar *src, int size)
{
  int i;

  for (i = 0; i < size; i++)
    dst[i] = src[i];
}

/**
 * @brief The main OpenCL kernel for texture compression.
 *
 * Each work-item in the NDRange is responsible for compressing a single 4x4
 * block of the source image.
 *
 * @param src Input image data in global memory.
 * @param dst Output buffer for compressed block data in global memory.
 * @param width Width of the source image.
 * @param height Height of the source image.
 * @param error An array in global memory to store the compression error for each block.
 */
__kernel void gpuCompressProcess(__global uchar *src,
                                __global uchar *dst,
                                int width, int height,
                                __global unsigned long *error)
{
  uint rows = (height+(4-height%4)) / 4;
  uint columns = width / 4;
  // Threading Model: Get a unique ID for each 4x4 block.
  uint gid = get_global_id(0);
  Color ver_blocks[16], hor_blocks[16];
  __global Color *row0, *row1, *row2, *row3;

  // Map the 1D global ID to a 2D block coordinate.
  uint row_id = gid / columns; 
  uint column_id = gid % columns;

  // Calculate memory offsets for the current block.
  uint src_offset = width * 4 * 4 * row_id + column_id * 16;
  uint dst_offset = 8 * gid;

  // Handle image boundaries where height is not a multiple of 4.
  if (row_id == rows-1) {
    if (height % 4 == 3) {
      row0 = (__global Color *) (src + src_offset);
      row1 = row0 + width;
      row2 = row1 + width;
      row3 = (__global Color *) (src + src_offset);
    } else if (height % 4 == 2) {
      row0 = (__global Color *) (src + src_offset);
      row1 = row0 + width;
      row2 = (__global Color *) (src + src_offset);
      row3 = (__global Color *) (src + src_offset);
    } else if (height % 4 == 1) {
      row0 = (__global Color *) (src + src_offset);
      row1 = (__global Color *) (src + src_offset);
      row2 = (__global Color *) (src + src_offset);
      row3 = (__global Color *) (src + src_offset);
    } else if (height % 4 == 0) {
      row0 = (__global Color *) (src + src_offset);
      row1 = row0 + width;
      row2 = row1 + width;
      row3 = row2 + width;
    }
  } else {
    row0 = (__global Color *) (src + src_offset);
    row1 = row0 + width;
    row2 = row1 + width;
    row3 = row2 + width;
  }

  // Gather pixel data for horizontal and vertical sub-block splits.
  my_memcpy((uchar *) ver_blocks, (__global uchar *) row0, 8);
	my_memcpy((uchar *) (ver_blocks + 2), (__global uchar *) row1, 8);
	my_memcpy((uchar *) (ver_blocks + 4), (__global uchar *) row2, 8);
	my_memcpy((uchar *) (ver_blocks + 6), (__global uchar *) row3, 8);
	my_memcpy((uchar *) (ver_blocks + 8), (__global uchar *) (row0 + 2), 8);
	my_memcpy((uchar *) (ver_blocks + 10), (__global uchar *) (row1 + 2), 8);
	my_memcpy((uchar *) (ver_blocks + 12), (__global uchar *) (row2 + 2), 8);
	my_memcpy((uchar *) (ver_blocks + 14), (__global uchar *) (row3 + 2), 8);
	
	my_memcpy((uchar *) hor_blocks, (__global uchar *) row0, 16);
	my_memcpy((uchar *) (hor_blocks + 4), (__global uchar *) row1, 16);
	my_memcpy((uchar *) (hor_blocks + 8), (__global uchar *) row2, 16);
	my_memcpy((uchar *) (hor_blocks + 12), (__global uchar *) row3, 16);

  // Call the main compression function and store the resulting error.
  error[gid] = compressBlock(dst + dst_offset, ver_blocks, hor_blocks, 0x7FFFFFFF);
}

// C++ Host Code Section

#include "compress.hpp"


using namespace std;

/**
 * @brief Constructor for the TextureCompressor class.
 *
 * Initializes the OpenCL environment by selecting a platform and GPU device,
 * and creating a context and command queue.
 */
TextureCompressor::TextureCompressor()
{
  cl_uint num_platforms;
  cl_int ret;

  this->platform_ids = new cl_platform_id[2];

  ret = clGetPlatformIDs(2, this->platform_ids, &num_platforms);
  assert(num_platforms == 2);
  assert(ret == CL_SUCCESS);

  ret = clGetDeviceIDs(this->platform_ids[0], CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  assert(ret == CL_SUCCESS);

  context = clCreateContext(0, 1, &device, NULL, NULL, &ret);
  assert(ret == CL_SUCCESS);
  command_queue = clCreateCommandQueue(context, device, 0, &ret);
  assert(ret == CL_SUCCESS);
}

/**
 * @brief Destructor for the TextureCompressor class.
 *
 * Releases the OpenCL context and frees allocated memory.
 */
TextureCompressor::~TextureCompressor()
{
  delete[] platform_ids;
  clReleaseContext(context);
}

/**
 * @brief Compresses a source image using the OpenCL kernel.
 *
 * This method orchestrates the entire compression process:
 * 1. Reads the OpenCL kernel source from the file system.
 * 2. Compiles and builds the OpenCL program.
 * 3. Allocates memory buffers on the GPU.
 * 4. Transfers input data from host to GPU.
 * 5. Executes the compression kernel.
 * 6. Transfers results (compressed data and errors) from GPU to host.
 * 7. Returns the total compression error.
 */
unsigned long TextureCompressor::compress(const uint8_t* src,
									  uint8_t* dst,
									  int width,
									  int height)
{
	const size_t global_size = ceil((float) (height * width) / 16);
  std::ifstream file;
  std::string source, line;
  cl_mem src_mem, dst_mem, error_mem;
  cl_int ret;
  unsigned long *errors;
  const char *source_str;

  // Read the kernel source from this same file.
  file.open("gpu_kernel.cl");
  if (file.is_open()) {
    while (file.good()) {
      getline(file, line);
      source.append(line + "
");
    }
  } else {
    assert(false);
  }
  file.close();

  source_str = source.c_str();

  // Create and build the OpenCL program.
  this->program = clCreateProgramWithSource(this->context,
                                            1,
                                            (const char **) &source_str,
                                            NULL,
                                            &ret);
  assert(ret == CL_SUCCESS);
  clBuildProgram(this->program, 1, &this->device, NULL, NULL, NULL);
  this->kernel = clCreateKernel(this->program,
                                "gpuCompressProcess",
                                &ret);
  assert(ret == CL_SUCCESS);
 
  // Create memory buffers on the device.
  src_mem = clCreateBuffer(this->context,
                            CL_MEM_COPY_HOST_PTR,
                            4 * width * height,
                            (void *) src,
                            &ret);
  assert(ret == CL_SUCCESS);
  dst_mem = clCreateBuffer(this->context,
                            CL_MEM_READ_WRITE,
                            width * height * 4 / 8,
                            (void *) NULL,
                            &ret);
  assert(ret == CL_SUCCESS);
  error_mem = clCreateBuffer(this->context,
                              CL_MEM_READ_WRITE,
                              global_size * sizeof(unsigned long),
                              (void *) NULL,
                              &ret);
  assert(ret == CL_SUCCESS);
  errors = (unsigned long *) malloc(sizeof(unsigned long) * global_size);

  // Set the arguments for the kernel.
  clSetKernelArg(this->kernel, 0, sizeof(cl_mem), (void *) &src_mem);
  clSetKernelArg(this->kernel, 1, sizeof(cl_mem), (void *) &dst_mem);
  clSetKernelArg(this->kernel, 2, sizeof(int), (void *) &width);
  clSetKernelArg(this->kernel, 3, sizeof(int), (void *) &height);
  clSetKernelArg(this->kernel, 4, sizeof(cl_mem), &error_mem);

  // Execute the kernel on the device.
  ret = clEnqueueNDRangeKernel(this->command_queue,
                                this->kernel,
                                1,
                                NULL,
                                &global_size, NULL, 0, NULL, NULL);
  assert(ret == CL_SUCCESS);
  // Read the results back from the device.
  ret = clEnqueueReadBuffer(this->command_queue,
                            error_mem,
                            CL_TRUE,
                            0,
                            sizeof(unsigned long) * global_size, 
                            errors, 0, NULL, NULL);


  assert(ret == CL_SUCCESS);
  ret = clEnqueueReadBuffer(this->command_queue,
                            dst_mem,
                            CL_TRUE,
                            0,
                            width * height * 4 / 8, 
                            dst, 0, NULL, NULL);
  assert(ret == CL_SUCCESS);

  // Sum up the errors from all blocks.
  unsigned long error_ret = 0;
  for (unsigned int i = 0; i < global_size; i++)
    error_ret += errors[i];

	return error_ret;
}
