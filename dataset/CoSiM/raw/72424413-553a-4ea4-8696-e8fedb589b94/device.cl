
#define uint8_t uchar
#define int8_t char
#define uint16_t ushort
#define int16_t short
#define uint32_t uint
#define int32_t int

#define UINT32_MAX UINT_MAX
#define INT32_MAX INT_MAX

typedef union {
	struct BgraColorType {
		uint8_t b;
		uint8_t g;
		uint8_t r;
		uint8_t a;
	} channels;
	uint8_t components[4];
	uint32_t bits;
} Color;

inline uint8_t round_to_5_bits(float val) {
	return clamp((uint8_t)(val * 31.0f / 255.0f + 0.5f), (uint8_t)0, (uint8_t)31);
}

inline uint8_t round_to_4_bits(float val) {
	return clamp((uint8_t)(val * 15.0f / 255.0f + 0.5f), (uint8_t)0, (uint8_t)15);
}



__constant static const int16_t g_codeword_tables[8][4] = {
	{-8, -2, 2, 8},
	{-17, -5, 5, 17},
	{-29, -9, 9, 29},
	{-42, -13, 13, 42},
	{-60, -18, 18, 60},
	{-80, -24, 24, 80},
	{-106, -33, 33, 106},
	{-183, -47, 47, 183}
};



__constant static const uint8_t g_mod_to_pix[4] = {3, 2, 0, 1};





















__constant static const uint8_t g_idx_to_num[4][8] = {
	{0, 4, 1, 5, 2, 6, 3, 7},        
	{8, 12, 9, 13, 10, 14, 11, 15},  
	{0, 4, 8, 12, 1, 5, 9, 13},      
	{2, 6, 10, 14, 3, 7, 11, 15}     
};


Color makeColor(Color base, int16_t lum) {
	int b = (int)(base.channels.b) + lum;
	int g = (int)(base.channels.g) + lum;
	int r = (int)(base.channels.r) + lum;
	Color color;
	color.channels.b = (uint8_t)(clamp(b, 0, 255));


	color.channels.g = (uint8_t)(clamp(g, 0, 255));
	color.channels.r = (uint8_t)(clamp(r, 0, 255));
	return color;
}



uint32_t getColorError(Color u, Color v) {
#ifdef USE_PERCEIVED_ERROR_METRIC
	float delta_b = (float)(u.channels.b) - v.channels.b;
	float delta_g = (float)(u.channels.g) - v.channels.g;
	float delta_r = (float)(u.channels.r) - v.channels.r;
	return (uint32_t)(0.299f * delta_b * delta_b +
								 0.587f * delta_g * delta_g +
								 0.114f * delta_r * delta_r);
#else
	int delta_b = (int)(u.channels.b) - v.channels.b;
	int delta_g = (int)(u.channels.g) - v.channels.g;
	int delta_r = (int)(u.channels.r) - v.channels.r;
	return delta_b * delta_b + delta_g * delta_g + delta_r * delta_r;
#endif
}



inline void WriteColors444(__global uint8_t* block,
						   Color color0,
						   Color color1) {
	
	block[0] = (color0.channels.r & 0xf0) | (color1.channels.r >> 4);
	block[1] = (color0.channels.g & 0xf0) | (color1.channels.g >> 4);
	block[2] = (color0.channels.b & 0xf0) | (color1.channels.b >> 4);
}

inline void WriteColors555(__global uint8_t* block,
						   Color color0,
						   Color color1) {
	
	const uint8_t two_compl_trans_table[8] = {
		4,  
		5,  
		6,  
		7,  
		0,  
		1,  
		2,  
		3,  
	};
	
	int16_t delta_r =
	(int16_t)(color1.channels.r >> 3) - (color0.channels.r >> 3);
	int16_t delta_g =
	(int16_t)(color1.channels.g >> 3) - (color0.channels.g >> 3);
	int16_t delta_b =
	(int16_t)(color1.channels.b >> 3) - (color0.channels.b >> 3);
	
	
	block[0] = (color0.channels.r & 0xf8) | two_compl_trans_table[delta_r + 4];
	block[1] = (color0.channels.g & 0xf8) | two_compl_trans_table[delta_g + 4];
	block[2] = (color0.channels.b & 0xf8) | two_compl_trans_table[delta_b + 4];
}

inline void WriteCodewordTable(__global uint8_t* block,
							   uint8_t sub_block_id,
							   uint8_t table) {
	
	uint8_t shift = (2 + (3 - sub_block_id * 3));
	block[3] &= ~(0x07 << shift);
	block[3] |= table << shift;
}

inline void WritePixelData(__global uint8_t* block, uint32_t pixel_data) {
	block[4] |= pixel_data >> 24;
	block[5] |= (pixel_data >> 16) & 0xff;
	block[6] |= (pixel_data >> 8) & 0xff;
	block[7] |= pixel_data & 0xff;
}

inline void WriteFlip(__global uint8_t* block, bool flip) {
	block[3] &= ~0x01;
	block[3] |= (uint8_t)(flip);
}

inline void WriteDiff(__global uint8_t* block, bool diff) {
	block[3] &= ~0x02;
	block[3] |= (uint8_t)(diff) << 1;
}





inline Color makeColor444(const float* bgr) {
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





inline Color makeColor555(const float* bgr) {
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
	
void getAverageColor(const Color* src, float* avg_color)
{
	uint32_t sum_b = 0, sum_g = 0, sum_r = 0;
	
	for (unsigned int i = 0; i < 8; ++i) {
		sum_b += src[i].channels.b;
		sum_g += src[i].channels.g;
		sum_r += src[i].channels.r;
	}
	
	const float kInv8 = 1.0f / 8.0f;
	avg_color[0] = (float)(sum_b) * kInv8;
	avg_color[1] = (float)(sum_g) * kInv8;
	avg_color[2] = (float)(sum_r) * kInv8;
}
	
unsigned long computeLuminance(__global uint8_t* block,
						   __private const Color* src,
						   Color base,
						   int sub_block_id,
						   __constant const uint8_t* idx_to_num_tab,
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
				Color color = candidate_color[mod_idx];
				
				uint32_t mod_err = getColorError(src[i], color);
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


bool tryCompressSolidBlock(__global uint8_t* dst,
						   __private const Color* src)
{
	for (unsigned int i = 1; i < 16; ++i) {
		if (src[i].bits != src[0].bits)
			return false;
	}
	
	
	for (int i = 0; i < 8; i++)
		dst[i] = 0;
	
	float src_color_float[3] = {(float)(src->channels.b),
		(float)(src->channels.g),
		(float)(src->channels.r)};
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
			Color color = makeColor(base, lum);
			
			uint32_t mod_err = getColorError(*src, color);
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
	return true;
}

void compress(__global uint8_t* dst,
		__private const Color* ver_src,
		__private const Color* hor_src,


		unsigned long threshold)
{
	if (tryCompressSolidBlock(dst, ver_src)) {
		return;
	}
	
	__private const Color* sub_block_src[4] = {ver_src, ver_src + 8, hor_src, hor_src + 8};
	
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
	
	
	
	
	uint32_t sub_block_err[4] = {0};
	for (unsigned int i = 0; i < 4; ++i) {
		for (unsigned int j = 0; j < 8; ++j) {
			sub_block_err[i] += getColorError(sub_block_avg[i], sub_block_src[i][j]);
		}
	}
	
	bool flip =
	sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1];
	
	
	for (int i = 0; i < 8; i++)
		dst[i] = 0;
	
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
	
	
	computeLuminance(dst, sub_block_src[sub_block_off_0],
								   sub_block_avg[sub_block_off_0], 0,
								   g_idx_to_num[sub_block_off_0],
								   threshold);
	
	computeLuminance(dst, sub_block_src[sub_block_off_1],
								   sub_block_avg[sub_block_off_1], 1,
								   g_idx_to_num[sub_block_off_1],
								   threshold);
}

__kernel void etc(int width, int height,
			__global Color* src,
			__global uchar* dst) {
	int row_index = get_global_id(0) * 4;
	int col_index = get_global_id(1) * 4;

	__global Color* row0 = src + row_index * width + col_index;
	__global Color* row1 = row0 + width;
	__global Color* row2 = row1 + width;
	__global Color* row3 = row2 + width;

	Color ver_blocks[16], hor_blocks[16];

	ver_blocks[0].bits = row0[0].bits;
	ver_blocks[1].bits = row0[1].bits;
	ver_blocks[2].bits = row1[0].bits;
	ver_blocks[3].bits = row1[1].bits;
	ver_blocks[4].bits = row2[0].bits;
	ver_blocks[5].bits = row2[1].bits;
	ver_blocks[6].bits = row3[0].bits;
	ver_blocks[7].bits = row3[1].bits;
	ver_blocks[8+0].bits = row0[2+0].bits;
	ver_blocks[8+1].bits = row0[2+1].bits;
	ver_blocks[8+2].bits = row1[2+0].bits;
	ver_blocks[8+3].bits = row1[2+1].bits;
	ver_blocks[8+4].bits = row2[2+0].bits;
	ver_blocks[8+5].bits = row2[2+1].bits;
	ver_blocks[8+6].bits = row3[2+0].bits;
	ver_blocks[8+7].bits = row3[2+1].bits;

	hor_blocks[0].bits = row0[0].bits;
	hor_blocks[1].bits = row0[1].bits;
	hor_blocks[2].bits = row0[2].bits;
	hor_blocks[3].bits = row0[3].bits;
	hor_blocks[4+0].bits = row1[0].bits;
	hor_blocks[4+1].bits = row1[1].bits;
	hor_blocks[4+2].bits = row1[2].bits;
	hor_blocks[4+3].bits = row1[3].bits;
	hor_blocks[8+0].bits = row2[0].bits;
	hor_blocks[8+1].bits = row2[1].bits;
	hor_blocks[8+2].bits = row2[2].bits;
	hor_blocks[8+3].bits = row2[3].bits;
	hor_blocks[12+0].bits = row3[0].bits;
	hor_blocks[12+1].bits = row3[1].bits;
	hor_blocks[12+2].bits = row3[2].bits;
	hor_blocks[12+3].bits = row3[3].bits;

	compress(dst + (row_index * width) / 2 + col_index * 2, ver_blocks, hor_blocks, INT32_MAX);
}
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

#include 

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
#include "compress.hpp"
#include "helper.hpp"

#include 
#include 
#include 
#include 
#include 

using namespace std;

void gpu_find(cl_device_id &device, 
		cl_platform_id* &platform_list,
		cl_device_id* &device_list)
{
	cl_platform_id platform;

	cl_uint platform_num = 0;
	platform_list = NULL;

	cl_uint device_num = 0;
	device_list = NULL;

	
	CL_ERR( clGetPlatformIDs(0, NULL, &platform_num));
	platform_list = new cl_platform_id[platform_num];
	DIE(platform_list == NULL, "alloc platform_list");

	
	CL_ERR( clGetPlatformIDs(platform_num, platform_list, NULL));

	
	for(uint platf=0; platf<platform_num; platf++)
	{
		
		platform = platform_list[platf];
		DIE(platform == 0, "platform selection");

		
		if(clGetDeviceIDs(platform, 
			CL_DEVICE_TYPE_GPU, 0, NULL, &device_num) == CL_DEVICE_NOT_FOUND) {
			device_num = 0;
			continue;
		}



		device_list = new cl_device_id[device_num];
		DIE(device_list == NULL, "alloc devices");

		
		CL_ERR( clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU,
			  device_num, device_list, NULL));

		device = device_list[0];
		break;
	}

	delete[] platform_list;
	delete[] device_list;
}

TextureCompressor::TextureCompressor() {
	gpu_find(device, platform_ids, device_ids);

	cl_int ret;

	string kernel_src;

	
	context = clCreateContext(0, 1, &device, NULL, NULL, &ret);
	CL_ERR( ret );

	
	command_queue = clCreateCommandQueue(context, device, 0, &ret);
	CL_ERR( ret );

	read_kernel("device.cl", kernel_src);
	const char* kernel_c_str = kernel_src.c_str();
	program = clCreateProgramWithSource(context, 1, &kernel_c_str, NULL, &ret);
	CL_ERR( ret );

	
	ret = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	CL_COMPILE_ERR( ret, program, device );

	
	kernel = clCreateKernel(program, "etc", &ret);
	CL_ERR( ret );
}

TextureCompressor::~TextureCompressor() {
	CL_ERR( clReleaseCommandQueue(command_queue) );
	CL_ERR( clReleaseContext(context) );
}
	
unsigned long TextureCompressor::compress(const uint8_t* src,
									  uint8_t* dst,
									  int width,
									  int height)
{
	cl_int ret;
	size_t src_size = width * height * 4;
	size_t dst_size = width * height * 4 / 8;

	cl_mem src_dev = clCreateBuffer(context, CL_MEM_READ_WRITE, src_size, NULL, &ret);
	CL_ERR( ret );

	cl_mem dst_dev = clCreateBuffer(context, CL_MEM_READ_WRITE, dst_size, NULL, &ret);
	CL_ERR( ret );



	CL_ERR( clEnqueueWriteBuffer(command_queue, src_dev, CL_TRUE, 0, src_size,
				src, 0, NULL, NULL) );

	
	CL_ERR( clSetKernelArg(kernel, 0, sizeof(cl_int), (void *)&width) );
	CL_ERR( clSetKernelArg(kernel, 1, sizeof(cl_int), (void *)&height) );


	CL_ERR( clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&src_dev) );
	CL_ERR( clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&dst_dev) );

	size_t globalSize[2] = {height / 4, width / 4};


	ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL,
		globalSize, NULL, 0, NULL, NULL);
	CL_ERR( ret );

	CL_ERR( clEnqueueReadBuffer(command_queue, dst_dev, CL_TRUE, 0, dst_size,
				dst, 0, NULL, NULL) );

	
	CL_ERR( clFinish(command_queue) );

	CL_ERR( clReleaseMemObject(src_dev) );
	CL_ERR( clReleaseMemObject(dst_dev) );

	return 0;
}
