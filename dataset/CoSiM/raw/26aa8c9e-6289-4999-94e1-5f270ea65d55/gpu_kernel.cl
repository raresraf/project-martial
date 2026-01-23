
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

__constant uchar g_mod_to_pix[4] = {3, 2, 0, 1};

__constant uchar g_idx_to_num[4][8] = {
	{0, 4, 1, 5, 2, 6, 3, 7},        
	{8, 12, 9, 13, 10, 14, 11, 15},  
	{0, 4, 8, 12, 1, 5, 9, 13},      
	{2, 6, 10, 14, 3, 7, 11, 15}     
};

int clamp_signed(int val, int min, int max) {
	return val  max ? max : val);
}

uint clamp_unsigned(uint val, uint min, uint max) {
	return val  max ? max : val);
}

uchar round_to_5_bits(float val) {
	return (uchar) clamp_unsigned((uint) (val * 31.0f / 255.0f + 0.5f), 0, 31);
}

uchar round_to_4_bits(float val) {
	return (uchar) clamp_unsigned((uint) (val * 15.0f / 255.0f + 0.5f), 0, 15);
}

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

uint getColorError(const Color *u, const Color *v) {
	int delta_b = (int) (u->channels.b) - v->channels.b;
	int delta_g = (int) (u->channels.g) - v->channels.g;
	int delta_r = (int) (u->channels.r) - v->channels.r;
	return delta_b * delta_b + delta_g * delta_g + delta_r * delta_r;
}

void WriteColors444(__global uchar *block,
						   const Color *color0,
						   const Color *color1) {
	
	block[0] = (color0->channels.r & 0xf0) | (color1->channels.r >> 4);
	block[1] = (color0->channels.g & 0xf0) | (color1->channels.g >> 4);
	block[2] = (color0->channels.b & 0xf0) | (color1->channels.b >> 4);
}

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

void WriteCodewordTable(__global uchar *block,
							   uchar sub_block_id,
							   uchar table) {

	uchar shift = (2 + (3 - sub_block_id * 3));
	block[3] &= ~(0x07 << shift);
	block[3] |= table << shift;
}

void WritePixelData(__global uchar* block, uint pixel_data) {
	block[4] |= pixel_data >> 24;
	block[5] |= (pixel_data >> 16) & 0xff;
	block[6] |= (pixel_data >> 8) & 0xff;
	block[7] |= pixel_data & 0xff;
}

void WriteFlip(__global uchar *block, int flip) {
	block[3] &= ~0x01;
	block[3] |= (uchar) flip;
}

void WriteDiff(__global uchar *block, int diff) {
	block[3] &= ~0x02;
	block[3] |= ((uchar) diff) << 1;
}

void ExtractBlock(__global uchar *dst, const uchar *src, int width) {
  int i, j;

	for (j = 0; j < 4; ++j) {
    for(i = 0; i < 16; i++)
		  dst[j * 4 * 4 + i] = src[i];
		src += width * 4;
	}
}

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

Color makeColor555(const float *bgr) {
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

	
	


	for (tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		
		
		Color candidate_color[4];  
		for (mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			candidate_color[mod_idx] = makeColor(base, lum);
		}
		
		uint tbl_err = 0;
		
		for (i = 0; i < 8; ++i) {
			
			
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
		
		if (tbl_err < best_tbl_err) {
			best_tbl_err = tbl_err;
			best_tbl_idx = tbl_idx;
			
			if (tbl_err == 0)
				break;  
		}
	}

	WriteCodewordTable(block, sub_block_id, best_tbl_idx);

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

int tryCompressSolidBlock(__global uchar *dst,
						   const Color *src,
						   unsigned long *error)
{


  unsigned int i, tbl_idx, mod_idx, j;

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

unsigned long compressBlock(__global uchar *dst,
												   const Color *ver_src,
												   const Color *hor_src,
												   unsigned long threshold)
{
	unsigned long solid_error = 0;
  unsigned int i, j, light_idx;
	if (tryCompressSolidBlock(dst, ver_src, &solid_error)) {
		return solid_error;
	}
	
	const Color *sub_block_src[4] = {ver_src, ver_src + 8, hor_src, hor_src + 8};
	
	Color sub_block_avg[4];
	int use_differential[2] = {1, 1};
	
	
	
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
	
	
	
	


	uint sub_block_err[4] = {0};
	for (i = 0; i < 4; ++i) {
		for (j = 0; j < 8; ++j) {
			sub_block_err[i] += getColorError(&sub_block_avg[i], &sub_block_src[i][j]);
		}
	}
	
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

void my_memcpy(uchar *dst, __global uchar *src, int size)
{
  int i;

  for (i = 0; i < size; i++)
    dst[i] = src[i];
}

__kernel void gpuCompressProcess(__global uchar *src,
                                __global uchar *dst,
                                int width, int height,
                                __global unsigned long *error)
{
  uint rows = (height+(4-height%4)) / 4;
  uint columns = width / 4;
	uint gid = get_global_id(0);
  Color ver_blocks[16], hor_blocks[16];
  __global Color *row0, *row1, *row2, *row3;

  uint row_id = gid / columns; 
  uint column_id = gid % columns;

	uint src_offset = width * 4 * 4 * row_id + column_id * 16;
  uint dst_offset = 8 * gid;

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

  error[gid] = compressBlock(dst + dst_offset, ver_blocks, hor_blocks, 0x7FFFFFFF);
}
#include "compress.hpp"


using namespace std;

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

TextureCompressor::~TextureCompressor()
{
  delete[] platform_ids;
  clReleaseContext(context);
}

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

  file.open("gpu_kernel.cl");
  if (file.is_open()) {
    while (file.good()) {
      getline(file, line);
      source.append(line + "\n");
    }
  } else {
    assert(false);
  }
  file.close();

  source_str = source.c_str();

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

  clSetKernelArg(this->kernel, 0, sizeof(cl_mem), (void *) &src_mem);
  clSetKernelArg(this->kernel, 1, sizeof(cl_mem), (void *) &dst_mem);
  clSetKernelArg(this->kernel, 2, sizeof(int), (void *) &width);
  clSetKernelArg(this->kernel, 3, sizeof(int), (void *) &height);
  clSetKernelArg(this->kernel, 4, sizeof(cl_mem), &error_mem);

  ret = clEnqueueNDRangeKernel(this->command_queue,
                                this->kernel,
                                1,
                                NULL,
                                &global_size, NULL, 0, NULL, NULL);
  assert(ret == CL_SUCCESS);
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

  unsigned long error_ret = 0;
  for (int i = 0; i < global_size; i++)
    error_ret += errors[i];

	return error_ret;
}
