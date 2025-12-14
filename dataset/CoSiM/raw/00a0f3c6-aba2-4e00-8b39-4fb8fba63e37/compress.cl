


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

int clamp1(int val, int min, int max) {
	return val  max ? max : val);
}



unsigned int clamp2(unsigned int val, unsigned int min, unsigned int max) {
	return val  max ? max : val);
}

unsigned char round_to_5_bits(float val) {
	return (unsigned char) clamp2(val * 31.0f / 255.0f + 0.5f, 0, 31);
}

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


unsigned int getColorError(const Color u, const Color v) {
#ifdef USE_PERCEIVED_ERROR_METRIC
	float delta_b = (float) u.channels.b - v.channels.b;
	float delta_g = (float) u.channels.g - v.channels.g;
	float delta_r = (float) u.channels.r - v.channels.r;
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

void WriteColors444(global unsigned char* block,
									 const Color color0,
									 const Color color1) {
	
	block[0] = (color0.channels.r & 0xf0) | (color1.channels.r >> 4);
	block[1] = (color0.channels.g & 0xf0) | (color1.channels.g >> 4);


	block[2] = (color0.channels.b & 0xf0) | (color1.channels.b >> 4);
}

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
	(short) (color1.channels.r >> 3) - (color0.channels.r >> 3);
	short delta_g =
	(short) (color1.channels.g >> 3) - (color0.channels.g >> 3);
	short delta_b =
	(short) (color1.channels.b >> 3) - (color0.channels.b >> 3);
	
	
	block[0] = (color0.channels.r & 0xf8) | two_compl_trans_table[delta_r + 4];


	block[1] = (color0.channels.g & 0xf8) | two_compl_trans_table[delta_g + 4];
	block[2] = (color0.channels.b & 0xf8) | two_compl_trans_table[delta_b + 4];
}

void WriteCodewordTable(global unsigned char* block,
							   unsigned char sub_block_id,
							   unsigned char table) {
	
	unsigned char shift = (2 + (3 - sub_block_id * 3));
	block[3] &= ~(0x07 << shift);
	block[3] |= table << shift;
}

void WritePixelData(global unsigned char* block, unsigned int pixel_data) {
	block[4] |= pixel_data >> 24;
	block[5] |= (pixel_data >> 16) & 0xff;
	block[6] |= (pixel_data >> 8) & 0xff;
	block[7] |= pixel_data & 0xff;
}

void WriteFlip(global unsigned char* block, char flip) {
	block[3] &= ~0x01;
	block[3] |= (unsigned char) flip;
}

void WriteDiff(global unsigned char* block, char diff) {
	block[3] &= ~0x02;
	block[3] |= (unsigned char) (diff) << 1;
}

inline void ExtractBlock(global unsigned char* dst, const unsigned char* src, int width) {
	int i,j;

	for (i = 0; i < 4; ++i) {
		int index = i * 4 * 4;
		for (j = 0; j < 16; ++j) {
			dst[index + j] = src[j];
		}
		src += width * 4;
	}
}





inline Color makeColor444(const float* bgr) {
	unsigned char b4 = round_to_4_bits(bgr[0]);
	unsigned char g4 = round_to_4_bits(bgr[1]);
	unsigned char r4 = round_to_4_bits(bgr[2]);
	Color bgr444;
	bgr444.channels.b = (b4 << 4) | b4;
	bgr444.channels.g = (g4 << 4) | g4;


	bgr444.channels.r = (r4 << 4) | r4;
	
	bgr444.channels.a = 0x44;
	return bgr444;
}





inline Color makeColor555(const float* bgr) {
	unsigned char b5 = round_to_5_bits(bgr[0]);
	unsigned char g5 = round_to_5_bits(bgr[1]);
	unsigned char r5 = round_to_5_bits(bgr[2]);
	Color bgr555;
	bgr555.channels.b = (b5 > 2);
	bgr555.channels.g = (g5 > 2);


	bgr555.channels.r = (r5 > 2);
	
	bgr555.channels.a = 0x55;
	return bgr555;
}
	
void getAverageColor(const Color* src, float* avg_color)
{
	unsigned int sum_b = 0, sum_g = 0, sum_r = 0, i;
	
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

	
	
	for (tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		
		
		Color candidate_color[4];  
		for (mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			candidate_color[mod_idx] = makeColor(base, lum);
		}
		
		unsigned int tbl_err = 0;
		
		for (i = 0; i < 8; ++i) {
			
			
			unsigned int best_mod_err = threshold;
			for (mod_idx = 0; mod_idx < 4; ++mod_idx) {
				const Color color = candidate_color[mod_idx];
				
				unsigned int mod_err = getColorError(src[i], color);
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

	unsigned int pix_data = 0;

	for (i = 0; i < 8; ++i) {
		unsigned char mod_idx = best_mod_idx[best_tbl_idx][i];
		unsigned char pix_idx = g_mod_to_pix[mod_idx];
		
		unsigned int lsb = pix_idx & 0x1;
		unsigned int msb = pix_idx >> 1;
		
		
		int texel_num = idx_to_num_tab[i];
		pix_data |= msb << (texel_num + 16);
		pix_data |= lsb << (texel_num);
	}

	WritePixelData(block, pix_data);

	return best_tbl_err;
}


bool tryCompressSolidBlock(global unsigned char* dst,
						   const Color* src,
						   unsigned long* error)
{
	unsigned int i, j;
	unsigned int tbl_idx;
	unsigned int mod_idx;

	for (i = 1; i < 16; ++i) {
		if (src[i].bits != src[0].bits)
			return false;
	}
	
	
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
	
	
	
	for (tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		
		
		for ( mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			const Color color = makeColor(base, lum);
			
			unsigned int mod_err = getColorError(*src, color);
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
	
	unsigned char pix_idx = g_mod_to_pix[best_mod_idx];
	unsigned int lsb = pix_idx & 0x1;
	unsigned int msb = pix_idx >> 1;
	
	unsigned int pix_data = 0;
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

unsigned long compressBlock(global unsigned char* dst,
												   const Color* ver_src,
												   const Color* hor_src,
												   unsigned long threshold)
{
	unsigned long solid_error = 0;
	unsigned int i, j;
	unsigned int light_idx;

	if (tryCompressSolidBlock(dst, ver_src, &solid_error)) {
		return solid_error;
	}
	
	const Color* sub_block_src[4] = {ver_src, ver_src + 8, hor_src, hor_src + 8};
	
	Color sub_block_avg[4];
	bool use_differential[2] = {true, true};
	
	
	
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
	
	
	
	
	unsigned int sub_block_err[4] = {0};
	for (i = 0; i < 4; ++i) {
		for (j = 0; j < 8; ++j) {
			sub_block_err[i] += getColorError(sub_block_avg[i], sub_block_src[i][j]);
		}
	}
	
	char flip =
	sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1];
	
	
	dst[0] = 0;
	dst[1] = 0;
	dst[2] = 0;
	dst[3] = 0;
	dst[4] = 0;
	dst[5] = 0;
	dst[6] = 0;
	dst[7] = 0;
	
	WriteDiff(dst, use_differential[!!flip]);
	WriteFlip(dst, flip);
	
	unsigned char sub_block_off_0 = flip ? 2 : 0;
	unsigned char sub_block_off_1 = sub_block_off_0 + 1;
	
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

__kernel void compress_kernel(int width, int height,
                              global unsigned int *compress_error,
															global unsigned char *src,
                              global unsigned char *dst)
{
 	unsigned int num_rows = height / 4, num_columns = width / 4;
	unsigned int gid = get_global_id(0);
	unsigned int i, j;
  unsigned int row_index = gid / num_columns; 
  unsigned int column_index = gid - row_index * num_columns;


	unsigned int src_index = 16*width*row_index + column_index*4*4;
  unsigned int dst_index = gid*4*2;
  
  Color ver_blocks[16], hor_blocks[16];
  global Color *row_start;

  row_start = (global Color *) &(src[src_index]);

	for (i = 0; i <= 6; i+=2) {


      ver_blocks[i].bits = row_start->bits;
      ver_blocks[i+1].bits = (row_start+1)->bits;
      row_start+=width;
	}

  row_start = (global Color *) &(src[src_index]);

	for (i = 8; i <= 14; i+=2) {


      ver_blocks[i].bits = (row_start+2)->bits;
      ver_blocks[i+1].bits = (row_start+3)->bits;
      row_start+=width;
	}

  row_start = (global Color *) &(src[src_index]);

  for (i = 0; i <= 12; i+=4) {


    hor_blocks[i].bits = row_start->bits;
    hor_blocks[i+1].bits = (row_start+1)->bits;
    hor_blocks[i+2].bits = (row_start+2)->bits;
    hor_blocks[i+3].bits = (row_start+3)->bits;

    row_start+=width;
  }

  atomic_add(compress_error, compressBlock(&dst[dst_index], ver_blocks, hor_blocks, 4294967295));
}
#include "compress.hpp"

#include 
#include 
#include 
#include 

using namespace std;

TextureCompressor::TextureCompressor() {
  cl_uint platforms;
  cl_int rc;

  platform_ids = (cl_platform_id *) malloc(sizeof(cl_platform_id) * 2);

  rc = clGetPlatformIDs(2, platform_ids, &platforms);
  if (rc != CL_SUCCESS) {
    printf("platform error\n");
    exit(1);
  }

  rc = clGetDeviceIDs(platform_ids[0], CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  if (rc != CL_SUCCESS) {
    printf("device id error\n");
    exit(1);
  }

  context = clCreateContext(0, 1, &device, NULL, NULL, &rc);
  if (rc != CL_SUCCESS) {
    printf("context error\n");
    exit(1);
  }

  command_queue = clCreateCommandQueue(context, device, 0, &rc);
  if (rc != CL_SUCCESS) {
    printf("queue error\n");
    exit(1);
  }
}

TextureCompressor::~TextureCompressor() {
  free(platform_ids);
  clReleaseContext(context);
}

unsigned long TextureCompressor::compress(const uint8_t* src,
									  uint8_t* dst,
									  int width,
									  int height)
{
	size_t dimension = (width / 4) * (height / 4);
  char *kernel_source = NULL;
  FILE *kernel_fp = NULL;
  char line_buf[300];
  unsigned int num_of_bytes = 0;
  cl_int rc;
  cl_mem cl_src, cl_dst, cl_compress_err;
  unsigned int compress_error;

  kernel_fp = fopen("compress.cl", "r");
  if (kernel_fp == NULL) {
    fprintf(stderr, "File not found\n");
    exit(1);
  }

  while(fgets(line_buf, 300, kernel_fp) != NULL) {
    num_of_bytes += strlen(line_buf);
  }

  kernel_source = (char *) malloc (num_of_bytes);
  strcpy(kernel_source, "");
  fclose(kernel_fp);

  kernel_fp = fopen("compress.cl", "r");
  if (kernel_fp == NULL) {
    fprintf(stderr, "File not found\n");
    exit(1);
  }

  while(fgets(line_buf, 300, kernel_fp) != NULL) {
    strcat(kernel_source, line_buf);
  }

  fclose(kernel_fp);

  program = clCreateProgramWithSource(context,
            1,
            (const char **) &kernel_source,
            NULL,
            &rc);
  if (rc != CL_SUCCESS) {
    printf("create program error\n");
    exit(1);
  }

  clBuildProgram(program, 1, &device, NULL, NULL, NULL);
  kernel = clCreateKernel(program,
            "compress_kernel",
            &rc);

  cl_src = clCreateBuffer(context,
            CL_MEM_COPY_HOST_PTR,
            4*width*height,
            (void *) src,
            &rc);
  if (rc != CL_SUCCESS) {
    printf("src buffer error\n");
    exit(1);
  }
  cl_dst = clCreateBuffer(context,
            CL_MEM_READ_WRITE,
            width*height*4/8,
            (void *) NULL,
            &rc);
  if (rc != CL_SUCCESS) {
    printf("dst buffer error\n");
    exit(1);
  }
  cl_compress_err = clCreateBuffer(context,
                    CL_MEM_READ_WRITE,
                    sizeof(unsigned int),
                    (void *) NULL,
                    &rc);
  if (rc != CL_SUCCESS) {
    printf("error buffer error\n");
    exit(1);
  }
  clSetKernelArg(kernel, 0, sizeof(int), &width);
  clSetKernelArg(kernel, 1, sizeof(int), &height);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &cl_compress_err);
  clSetKernelArg(kernel, 3, sizeof(cl_mem), &cl_src);


  clSetKernelArg(kernel, 4, sizeof(cl_mem), &cl_dst);

  rc = clEnqueueNDRangeKernel(command_queue,
                              kernel,
                              1,
                              NULL,
                              &dimension,
                              NULL,
                              0,
                              NULL,
                              NULL);
  if (rc != CL_SUCCESS) {
    printf("enqueue error\n");
    exit(1);
  }



  rc = clEnqueueReadBuffer(this->command_queue,
                            cl_dst,
                            CL_TRUE,
                            0,
                            width*height*4/8, 
                            dst,
                            0,
                            NULL,
                            NULL);
  if (rc != CL_SUCCESS) {
    printf("read dst error\n");
    exit(1);
  }

  rc = clEnqueueReadBuffer(command_queue,
                            cl_compress_err,
                            CL_TRUE,
                            0,
                            sizeof(unsigned int), 
                            &compress_error,
                            0,
                            NULL,
                            NULL);
  if (rc != CL_SUCCESS) {
    printf("read error buffer error\n");
    exit(1);
  }

  free(kernel_source);
	return compress_error;
}
