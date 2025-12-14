
#define B_V 0
#define G_V 1
#define R_V 2
#define A_V 3

typedef union color {
	uchar channels[4];
	uint bits;
} Color;

__constant __attribute__((aligned(16))) short g_codeword_tables[8][4] = {
	{-8, -2, 2, 8},
	{-17, -5, 5, 17},
	{-29, -9, 9, 29},
	{-42, -13, 13, 42},
	{-60, -18, 18, 60},
	{-80, -24, 24, 80},
	{-106, -33, 33, 106},
	{-183, -47, 47, 183}};

__constant uchar g_mod_to_pix[4] = {3, 2, 0, 1};
__constant uchar g_idx_to_num[4][8] = {
	{0, 4, 1, 5, 2, 6, 3, 7},        
	{8, 12, 9, 13, 10, 14, 11, 15},  


	{0, 4, 8, 12, 1, 5, 9, 13},      
	{2, 6, 10, 14, 3, 7, 11, 15}     
};

void *
memcpy_gpu (void *dest, __global void *src, size_t len)
{
  char *d = (char*)dest;
  __global const char *s = (__global char*)src;
  while (len--)
    *d++ = *s++;
  return dest;
}

int my_clampI(int val, int min, int max) {
	return val  max ? max : val);
}

float my_clampF(float val, float min, float max) {
	return val  max ? max : val);
}

uchar my_clampC(uchar val, uchar min, uchar max) {
	return val  max ? max : val);
}

uchar round_to_5_bits(float val) {
	return my_clampC(val * 31.0f / 255.0f + 0.5f, 0, 31);
}

uchar round_to_4_bits(float val) {
	return my_clampC(val * 15.0f / 255.0f + 0.5f, 0, 15);
}

Color makeColor(const Color* base, short lum) {
	int b = (int)(base->channels[B_V]) + lum;
	int g = (int)(base->channels[G_V]) + lum;
	int r = (int)(base->channels[R_V]) + lum;
	Color color;
	color.channels[B_V] = (uchar)(my_clampI(b, 0, 255));
	color.channels[G_V] = (uchar)(my_clampI(g, 0, 255));


	color.channels[R_V] = (uchar)(my_clampI(r, 0, 255));
	return color;
}

uint getColorError(const Color* u, const Color* v) {
#ifdef USE_PERCEIVED_ERROR_METRIC
	float delta_b = (float)(u->channels.[B_V]) - v->channels[B_V]
	float delta_g = (float)(u->channels.[G_V]) - v->channels[G_V]
	float delta_r = (float)(u->channels.[R_V]) - v->channels[R_V]



	return (uint)(0.299f * delta_b * delta_b +
								 0.587f * delta_g * delta_g +
								 0.114f * delta_r * delta_r);
#else
	int delta_b = (int)(u->channels[B_V]) - v->channels[B_V];
	int delta_g = (int)(u->channels[G_V]) - v->channels[G_V];
	int delta_r = (int)(u->channels[R_V]) - v->channels[R_V];

	return delta_b * delta_b + delta_g * delta_g + delta_r * delta_r;
#endif
}

void WriteColors444(__global uchar* block,
						   const Color* color0,
						   const Color* color1) {
	
	block[0] = (color0->channels[R_V] & 0xf0) | (color1->channels[R_V] >> 4);
	block[1] = (color0->channels[G_V] & 0xf0) | (color1->channels[G_V] >> 4);
	block[2] = (color0->channels[B_V] & 0xf0) | (color1->channels[B_V] >> 4);
}



void WriteColors555(__global uchar* block,
						   const Color* color0,
						   const Color* color1) {
	
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
	(short)(color1->channels[R_V] >> 3) - (color0->channels[R_V] >> 3);
	short delta_g =
	(short)(color1->channels[G_V] >> 3) - (color0->channels[G_V] >> 3);
	short delta_b =
	(short)(color1->channels[B_V] >> 3) - (color0->channels[B_V] >> 3);
	
	
	block[0] = (color0->channels[R_V] & 0xf8) | two_compl_trans_table[delta_r + 4];
	block[1] = (color0->channels[G_V] & 0xf8) | two_compl_trans_table[delta_g + 4];
	block[2] = (color0->channels[B_V] & 0xf8) | two_compl_trans_table[delta_b + 4];
}

void WriteCodewordTable(__global uchar* block,
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

void WriteFlip(__global uchar* block, bool flip) {
	block[3] &= ~0x01;
	block[3] |= (uchar)flip;
}

void WriteDiff(__global uchar* block, bool diff) {
	block[3] &= ~0x02;
	block[3] |= (uchar)(diff) << 1;
}

Color makeColor444(const float* bgr) {
	uchar b4 = round_to_4_bits(bgr[0]);
	uchar g4 = round_to_4_bits(bgr[1]);
	uchar r4 = round_to_4_bits(bgr[2]);
	Color bgr444;
	bgr444.channels[B_V] = (b4 << 4) | b4;
	bgr444.channels[G_V] = (g4 << 4) | g4;
	bgr444.channels[R_V] = (r4 << 4) | r4;
	
	bgr444.channels[A_V] = 0x44;
	return bgr444;
}

Color makeColor555(const float bgr[3]) {
	uchar b5 = round_to_5_bits(bgr[0]);
	uchar g5 = round_to_5_bits(bgr[1]);
	uchar r5 = round_to_5_bits(bgr[2]);
	Color bgr555;
	bgr555.channels[B_V] = (b5 > 2);
	bgr555.channels[G_V] = (g5 > 2);
	bgr555.channels[R_V] = (r5 > 2);
	
	bgr555.channels[A_V] = 0x55;
	return bgr555;
}

void getAverageColor(const Color* src, float* avg_color)
{
	uint sum_b = 0, sum_g = 0, sum_r = 0;
	
	for (unsigned int i = 0; i < 8; ++i) {
		sum_b += src[i].channels[B_V];
		sum_g += src[i].channels[G_V];
		sum_r += src[i].channels[R_V];
	}
	
	const float kInv8 = 1.0f / 8.0f;
	avg_color[0] = (float)(sum_b) * kInv8;
	avg_color[1] = (float)(sum_g) * kInv8;
	avg_color[2] = (float)(sum_r) * kInv8;
}


unsigned long computeLuminance(__global uchar* block,
						   const Color* src,
						   const Color* base,
						   int sub_block_id,
						   __constant const uchar* idx_to_num_tab,
						   unsigned long threshold)
{
	uint best_tbl_err = threshold;
	uchar best_tbl_idx = 0;
	uchar best_mod_idx[8][8];  

	
	
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		
		
		Color candidate_color[4];  
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			candidate_color[mod_idx] = makeColor(base, lum);
		}
		
		uint tbl_err = 0;
		for (unsigned int i = 0; i < 8; ++i) {
			
			
			uint best_mod_err = threshold;
			for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
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

int tryCompressSolidBlock(__global uchar* dst,
						   const Color* src,
						   unsigned long* error)
{
	for (unsigned int i = 1; i < 16; ++i) {
		if (src[i].bits != src[0].bits)
			return false;
	}
	
	
	for (int i = 0; i < 8; i++)
		dst[i] = 0; 

	float src_color_float[3] = {(float)(src->channels[B_V]),
		(float)(src->channels[G_V]),
		(float)(src->channels[R_V])};

	Color base = makeColor555(src_color_float);
	
	WriteDiff(dst, 1);
	WriteFlip(dst, 0);
	WriteColors555(dst, &base, &base);
	
	uchar best_tbl_idx = 0;
	uchar best_mod_idx = 0;
	uint best_mod_err = 0xffffffff; 
	
	
	
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		
		
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {


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
	for (unsigned int i = 0; i < 2; ++i) {
		for (unsigned int j = 0; j < 8; ++j) {
			
			int texel_num = g_idx_to_num[i][j];
			pix_data |= msb << (texel_num + 16);
			pix_data |= lsb << (texel_num);
		}
	}
	
	WritePixelData(dst, pix_data);
	*error = 16 * best_mod_err;
	return 1;
}


unsigned long compressBlock(__global uchar* dst,
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
	ushort use_differential[2] = {true, true};
	
	
	
	for (unsigned int i = 0, j = 1; i < 4; i += 2, j += 2) {
		float avg_color_0[3];
		getAverageColor(sub_block_src[i], avg_color_0);
		Color avg_color_555_0 = makeColor555(avg_color_0);
		
		float avg_color_1[3];
		getAverageColor(sub_block_src[j], avg_color_1);
		Color avg_color_555_1 = makeColor555(avg_color_1);
		
		for (unsigned int light_idx = 0; light_idx < 3; ++light_idx) {
			int u = avg_color_555_0.channels[light_idx] >> 3;
			int v = avg_color_555_1.channels[light_idx] >> 3;
			
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
	for (unsigned int i = 0; i < 4; ++i) {
		for (unsigned int j = 0; j < 8; ++j) {
			sub_block_err[i] += getColorError(&sub_block_avg[i], &sub_block_src[i][j]);
		}
	}
	
	bool flip =
	sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1];
	
	
	for (int i = 0; i < 8; i++)
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


__kernel void compress(const int width,
                const int height,
				__global uchar* src,
				__global uchar* dst)
{
	int y = get_global_id(0);
	int x = get_global_id(1);
    Color ver_blocks[16];
    Color hor_blocks[16];
	
	src += width * 4 * 4 * y;
    dst += (8 * width/4 * y + 8 * x);
    y = y*4;
    x = x*4;

	__global Color* row0 = (__global Color*)(src + x * 4);
	__global Color* row1 = row0 + width;
	__global Color* row2 = row1 + width;
	__global Color* row3 = row2 + width;

	memcpy_gpu(ver_blocks, row0, 8);
	memcpy_gpu(ver_blocks + 2, row1, 8);
	memcpy_gpu(ver_blocks + 4, row2, 8);
	memcpy_gpu(ver_blocks + 6, row3, 8);
	memcpy_gpu(ver_blocks + 8, row0 + 2, 8);
	memcpy_gpu(ver_blocks + 10, row1 + 2, 8);
	memcpy_gpu(ver_blocks + 12, row2 + 2, 8);
	memcpy_gpu(ver_blocks + 14, row3 + 2, 8);
	
	memcpy_gpu(hor_blocks, row0, 16);


	memcpy_gpu(hor_blocks + 4, row1, 16);
	memcpy_gpu(hor_blocks + 8, row2, 16);
	memcpy_gpu(hor_blocks + 12, row3, 16);
	
	compressBlock(dst, ver_blocks, hor_blocks, 2147483647);
}
#ifndef _HELPERS_H_
#define _HELPERS_H_

#if __APPLE__
#include 
#else
#include 
#endif

#include 
#include 
#include 
#include 
#include 

using namespace std;

#define DIE(assertion, call_description)                    \
do {                                                        \
    if (assertion) {                                        \
            fprintf(stderr, "(%d): ",                       \
                            __LINE__);                      \
            perror(call_description);                       \
            exit(EXIT_FAILURE);                             \
    }                                                       \
} while(0);





inline void cl_get_compiler_err_log(cl_program program,
                             cl_device_id device)
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



inline const char* cl_get_string_err(cl_int err) {
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

inline int CL_ERR(int cl_ret)
{
	if(cl_ret != CL_SUCCESS){
		cout << endl << cl_get_string_err(cl_ret) << endl;
		return 1;
	}
	return 0;
}

inline int CL_COMPILE_ERR(int cl_ret,
                  cl_program program,
                  cl_device_id device)
{
	if(cl_ret != CL_SUCCESS){
		cout << endl << cl_get_string_err(cl_ret) << endl;
		cl_get_compiler_err_log(program, device);
		return 1;
	}
	return 0;
}


inline void read_kernel(string file_name, string &str_kernel)
{
	ifstream in_file(file_name.c_str());
	in_file.open(file_name.c_str());
	DIE( !in_file.is_open(), "ERR OpenCL kernel file. Same directory as binary ?" );

	stringstream str_stream;
	str_stream << in_file.rdbuf();

	str_kernel = str_stream.str();
}


#endif>>>> file: texture_compress_skl.cpp
#include "compress.hpp"
#include "helpers.hpp"
#include 
#include 
#include 
#include 
#include 

using namespace std;

static void gpu_find(cl_device_id &device);
static void gpu_run_kernel(cl_device_id device_id, 
						const uint8_t* src,
						uint8_t* dst,
						int width,
						int height);

TextureCompressor::TextureCompressor()
{
	gpu_find(this->device);
}

TextureCompressor::~TextureCompressor()
{

}
	


unsigned long TextureCompressor::compress(const uint8_t* src,
									  uint8_t* dst,
									  int width,
									  int height)
{
	gpu_run_kernel(this->device, src, dst, width, height);
	return 0;
}


void gpu_find(cl_device_id &device)
{
	uint platform_select = 1;
	uint device_select = 0;
	cl_platform_id platform;
	cl_uint platform_num = 0;
	cl_platform_id* platform_list = NULL;
	bool found = false;

	cl_uint device_num = 0;
	cl_device_id* device_list = NULL;
	
	size_t attr_size = 0;
	cl_char* attr_data = NULL;
	
	
	CL_ERR( clGetPlatformIDs(0, NULL, &platform_num));
	platform_list = new cl_platform_id[platform_num];
	DIE(platform_list == NULL, "alloc platform_list");
	
	
	CL_ERR( clGetPlatformIDs(platform_num, platform_list, NULL));
	
	
	for(uint platf=0; platf < platform_num; platf++)
	{
		
		
		platform = platform_list[platf];
		DIE(platform == 0, "platform selection");
		
		
		CL_ERR( clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &device_num));
		device_list = new cl_device_id[device_num];
		DIE(device_list == NULL, "alloc devices");
		
		
		CL_ERR( clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU,
							   device_num, device_list, NULL));
		
		
		if (device_num > 0) {
			device = device_list[0];
			break;
		}
	}
	


	delete[] platform_list;
	delete[] device_list;
}

void gpu_run_kernel(cl_device_id device_id, 
						const uint8_t* src,
						uint8_t* dst,
						int width,
						int height)
{
	size_t			global[2];
	size_t			local[2];
	string			kernel_src;
	cl_context		context;
	cl_command_queue commands;
	cl_program		program;
	cl_kernel		kernel;
	cl_uint			nd;
	cl_mem			a_in;
	cl_mem			b_out;
	int				i, ret;

	


	context = clCreateContext(0, 1, &device_id, NULL, NULL, &ret);
	CL_ERR( ret );
	
	commands = clCreateCommandQueue(context, device_id,
									CL_QUEUE_PROFILING_ENABLE, &ret);
	CL_ERR( ret );
	
	
	
	
	a_in   = clCreateBuffer(context,  CL_MEM_READ_WRITE,
							sizeof(Color) * width * height, NULL, NULL);
	b_out   = clCreateBuffer(context,  CL_MEM_READ_WRITE,
							width * height * 4 / 8, NULL, NULL);
	
	
	read_kernel("compress.cl", kernel_src);
	const char* kernel_c_str = kernel_src.c_str();
	
	
	program = clCreateProgramWithSource(context, 1,
				(const char **) &kernel_c_str, NULL, &ret);
	CL_ERR( ret );
	
	
	ret = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	CL_COMPILE_ERR( ret, program, device_id );
	
	
	kernel = clCreateKernel(program, "compress", &ret);
	CL_ERR( ret );
	
	
	ret  = 0;
	ret  = clSetKernelArg(kernel, 0, sizeof(int), &width);
	ret |= clSetKernelArg(kernel, 1, sizeof(int), &height);
	ret |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &a_in);
	ret |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &b_out);
	
	ret = clEnqueueWriteBuffer(commands, a_in, CL_TRUE, 0,
							   sizeof(Color) * height * width, src, 0, NULL, NULL);
	cl_event prof_event;
	
	global[0] =(size_t) height / 4;
	global[1] =(size_t) width / 4;
	local[0] =(size_t) 32;
	local[1] =(size_t) 32;
	
	ret = clEnqueueNDRangeKernel(commands, kernel, 2, NULL,
								 global, NULL, 0, NULL, NULL);
	
	CL_ERR(ret);
	clFinish(commands);
	
	ret = clEnqueueReadBuffer( commands, b_out, CL_TRUE, 0,
							  width * height * 4 / 8, dst, 0, NULL, NULL );
	
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseMemObject(a_in);
	clReleaseMemObject(b_out);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);
}