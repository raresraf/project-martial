




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

unsigned int my_clamp(unsigned int val, unsigned int min, unsigned int max) {
	return val  max ? max : val);
}

unsigned char round_to_5_bits(float val) {
	return (unsigned char) my_clamp(val * 31.0f / 255.0f + 0.5f, 0, 31);
}

unsigned char round_to_4_bits(float val) {
	return (unsigned char) my_clamp(val * 15.0f / 255.0f + 0.5f, 0, 15);
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
	color.channels.b = (unsigned char) my_clamp(b, 0, 255);
	color.channels.g = (unsigned char) my_clamp(g, 0, 255);
	color.channels.r = (unsigned char) my_clamp(r, 0, 255);
	return color;
}



unsigned int getColorError(const Color u, const Color v) {
	float delta_b = (float) u.channels.b - v.channels.b;


	float delta_g = (float) u.channels.g - v.channels.g;
	float delta_r = (float) u.channels.r - v.channels.r;
	return (unsigned int) (0.299f * delta_b * delta_b +
								 0.587f * delta_g * delta_g +
								 0.114f * delta_r * delta_r);
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

__kernel void compress_GPU(int width, int height,
	                	global unsigned int *compress_error,
						global unsigned char *src,
	                	global unsigned char *dst)
{
	unsigned int i;
	unsigned int gid = get_global_id(0);
	unsigned int row = gid / width / 4; 
	unsigned int column = gid - row * width / 4;
	unsigned int si = 16 * width * row + column * 4 * 4;
	unsigned int di = gid * 4 * 2;

	Color ver_blocks[16], hor_blocks[16];
	global Color *start;

	start = (global Color *) &(src[si]);
	i = 0;

	while (i <= 6) {
		ver_blocks[i].bits = start->bits;
		ver_blocks[i + 1].bits = (start + 1)->bits;
		
		start += width;
		i += 2;
	}

	start = (global Color *) &(src[si]);
	i = 8;

	while (i <= 14) {
		ver_blocks[i].bits = (start + 2)->bits;
		ver_blocks[i + 1].bits = (start + 3)->bits;
		
		start += width;
		i += 2;
	}

	start = (global Color *) &(src[si]);
	i = 0;

	while (i <= 12) {
		hor_blocks[i].bits = start->bits;
		hor_blocks[i + 1].bits = (start + 1)->bits;
		hor_blocks[i + 2].bits = (start + 2)->bits;
		hor_blocks[i + 3].bits = (start + 3)->bits;

		start += width;
		i += 4;
	}

	atomic_add(compress_error, compressBlock(&dst[di], ver_blocks, hor_blocks, 4294967295));
}>>>> file: helper.cpp
#include 
#include 
#include 
#include 
#include 

#include "helper.hpp"

using namespace std;


int CL_ERR(int cl_ret)
{
	if(cl_ret != CL_SUCCESS){
		cout << endl << cl_get_string_err(cl_ret) << endl;
		return 1;
	}
	return 0;
}


int CL_COMPILE_ERR(int cl_ret, cl_program program, cl_device_id device)
{
	if(cl_ret != CL_SUCCESS){
		cout << endl << cl_get_string_err(cl_ret) << endl;
		cl_get_compiler_err_log(program, device);
		return 1;
	}
	return 0;
}


void read_kernel(string file_name, string &str_kernel)
{
	ifstream in_file(file_name.c_str());
	in_file.open(file_name.c_str());
	DIE( !in_file.is_open(), "ERR OpenCL kernel file. Same directory as binary ?" );

	stringstream str_stream;
	str_stream << in_file.rdbuf();

	str_kernel = str_stream.str();
}


const char* cl_get_string_err(cl_int err) {
switch (err) {
  case CL_SUCCESS:                     	return  "Success!";
  case CL_DEVICE_NOT_FOUND:               return  "Device not found.";
  case CL_DEVICE_NOT_AVAILABLE:           return  "Device not available";
  case CL_COMPILER_NOT_AVAILABLE:         return  "Compiler not available";
  case CL_MEM_OBJECT_ALLOCATION_FAILURE:  return  "Memory object alloc fail";
  case CL_OUT_OF_RESOURCES:               return  "Out of resources";
  case CL_OUT_OF_HOST_MEMORY:             return  "Out of host memory";
  case CL_PROFILING_INFO_NOT_AVAILABLE:   return  "Profiling information N/A";
  case CL_MEM_COPY_OVERLAP:               return  "Memory copy overlap";
  case CL_IMAGE_FORMAT_MISMATCH:          return  "Image format mismatch";
  case CL_IMAGE_FORMAT_NOT_SUPPORTED:     return  "Image format no support";
  case CL_BUILD_PROGRAM_FAILURE:          return  "Program build failure";
  case CL_MAP_FAILURE:                    return  "Map failure";
  case CL_INVALID_VALUE:                  return  "Invalid value";
  case CL_INVALID_DEVICE_TYPE:            return  "Invalid device type";
  case CL_INVALID_PLATFORM:               return  "Invalid platform";
  case CL_INVALID_DEVICE:                 return  "Invalid device";
  case CL_INVALID_CONTEXT:                return  "Invalid context";
  case CL_INVALID_QUEUE_PROPERTIES:       return  "Invalid queue properties";
  case CL_INVALID_COMMAND_QUEUE:          return  "Invalid command queue";
  case CL_INVALID_HOST_PTR:               return  "Invalid host pointer";
  case CL_INVALID_MEM_OBJECT:             return  "Invalid memory object";
  case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:return  "Invalid image format desc";
  case CL_INVALID_IMAGE_SIZE:             return  "Invalid image size";
  case CL_INVALID_SAMPLER:                return  "Invalid sampler";
  case CL_INVALID_BINARY:                 return  "Invalid binary";
  case CL_INVALID_BUILD_OPTIONS:          return  "Invalid build options";
  case CL_INVALID_PROGRAM:                return  "Invalid program";
  case CL_INVALID_PROGRAM_EXECUTABLE:     return  "Invalid program exec";
  case CL_INVALID_KERNEL_NAME:            return  "Invalid kernel name";
  case CL_INVALID_KERNEL_DEFINITION:      return  "Invalid kernel definition";
  case CL_INVALID_KERNEL:                 return  "Invalid kernel";
  case CL_INVALID_ARG_INDEX:              return  "Invalid argument index";
  case CL_INVALID_ARG_VALUE:              return  "Invalid argument value";
  case CL_INVALID_ARG_SIZE:               return  "Invalid argument size";
  case CL_INVALID_KERNEL_ARGS:            return  "Invalid kernel arguments";
  case CL_INVALID_WORK_DIMENSION:         return  "Invalid work dimension";
  case CL_INVALID_WORK_GROUP_SIZE:        return  "Invalid work group size";
  case CL_INVALID_WORK_ITEM_SIZE:         return  "Invalid work item size";
  case CL_INVALID_GLOBAL_OFFSET:          return  "Invalid global offset";
  case CL_INVALID_EVENT_WAIT_LIST:        return  "Invalid event wait list";
  case CL_INVALID_EVENT:                  return  "Invalid event";
  case CL_INVALID_OPERATION:              return  "Invalid operation";
  case CL_INVALID_GL_OBJECT:              return  "Invalid OpenGL object";
  case CL_INVALID_BUFFER_SIZE:            return  "Invalid buffer size";
  case CL_INVALID_MIP_LEVEL:              return  "Invalid mip-map level";
  default:                                return  "Unknown";
  }
}


void cl_get_compiler_err_log(cl_program program, cl_device_id device)
{
	char* build_log;
	size_t log_size;

	
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
						  0, NULL, &log_size);
	build_log = new char[ log_size + 1 ];

	
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
						  log_size, build_log, NULL);
	build_log[ log_size ] = '\0';
	cout << endl << build_log << endl;
}
#ifndef CL_HELPER_H
#define CL_HELPER_H

#if __APPLE__
   #include 
#else
   #include 
#endif

using namespace std;

int CL_ERR(int cl_ret);
int CL_COMPILE_ERR(int cl_ret, cl_program program, cl_device_id device);

const char* cl_get_string_err(cl_int err);
void cl_get_compiler_err_log(cl_program program, cl_device_id device);

void read_kernel(string file_name, string &str_kernel);

#define DIE(assertion, call_description)  \
do { \
	if (assertion) { \
		fprintf(stderr, "(%d): ", __LINE__); \
		perror(call_description); \
		exit(EXIT_FAILURE); \
	} \
} while(0);

#endif




#include 
#include 
#include 
#include 
#include 

#include "compress.hpp"
#include "helper.hpp"

using namespace std;

TextureCompressor::TextureCompressor() {
  cl_uint platforms;
  cl_int rc;

  platform_ids = (cl_platform_id *) malloc(sizeof(cl_platform_id) * 2);
  DIE(platform_ids == NULL, "platform_ids error");

  CL_ERR(clGetPlatformIDs(2, platform_ids, &platforms));
  
  CL_ERR(clGetDeviceIDs(platform_ids[0], CL_DEVICE_TYPE_GPU, 1, &device, NULL));
  
  context = clCreateContext(0, 1, &device, NULL, NULL, &rc);
  CL_ERR(rc);

  command_queue = clCreateCommandQueue(context, device, 0, &rc);
  CL_ERR(rc);
}

TextureCompressor::~TextureCompressor() {
  free(platform_ids);
  CL_ERR(clReleaseCommandQueue(command_queue));
  CL_ERR(clReleaseContext(context));
}

unsigned long TextureCompressor::compress(const uint8_t* src, uint8_t* dst,
                    int width, int height)
{
  cl_int rc;
  cl_mem cl_src, cl_dst, cl_compress_err;
  size_t dimension = (width / 4) * (height / 4);
  unsigned int compress_error;

  string kernel_src;

  read_kernel("compress.cl", kernel_src);
  const char* kernel_c_str = kernel_src.c_str();

  program = clCreateProgramWithSource(context, 1,
            (const char **) &kernel_c_str, NULL, &rc);
  CL_ERR(rc);

  rc = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
  CL_COMPILE_ERR(rc, program, device);

  kernel = clCreateKernel(program, "compress_GPU", &rc);
  CL_ERR(rc);

  cl_src = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
            4 * width * height, (void *) src, &rc);
  CL_ERR(rc);

  cl_dst = clCreateBuffer(context, CL_MEM_READ_WRITE,
            width * height * 4 / 8, (void *) NULL, &rc);
  CL_ERR(rc);

  cl_compress_err = clCreateBuffer(context, CL_MEM_READ_WRITE,
                    sizeof(unsigned int), (void *) NULL, &rc);
  CL_ERR(rc);

  CL_ERR(clSetKernelArg(kernel, 0, sizeof(int), &width));
  CL_ERR(clSetKernelArg(kernel, 1, sizeof(int), &height));
  CL_ERR(clSetKernelArg(kernel, 2, sizeof(cl_mem), &cl_compress_err));
  CL_ERR(clSetKernelArg(kernel, 3, sizeof(cl_mem), &cl_src));
  CL_ERR(clSetKernelArg(kernel, 4, sizeof(cl_mem), &cl_dst));

  rc = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
                              &dimension, NULL, 0, NULL, NULL);
  CL_ERR(rc);

  rc = clEnqueueReadBuffer(this->command_queue, cl_dst, CL_TRUE, 0,
                            width * height * 4 / 8,  dst, 0,
                            NULL, NULL);
  CL_ERR(rc);

  rc = clEnqueueReadBuffer(command_queue, cl_compress_err, CL_TRUE,
                            0, sizeof(unsigned int), &compress_error,
                            0, NULL, NULL);
  CL_ERR(rc);

  return compress_error;
}
