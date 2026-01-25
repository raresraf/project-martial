
>>>> file: helper.cpp
#include 
#include 
#include 
#include 
#include 

#include "helper.hpp"

using namespace std;

/**
 * User/host function, check OpenCL function return code
 */
int CL_ERR(int cl_ret)
{
	if(cl_ret != CL_SUCCESS){
		cout << endl << cl_get_string_err(cl_ret) << endl;
		return 1;
	}
	return 0;
}

/**
 * User/host function, check OpenCL compilation return code
 */
int CL_COMPILE_ERR(int cl_ret, cl_program program, cl_device_id device)
{
	if(cl_ret != CL_SUCCESS){
		cout << endl << cl_get_string_err(cl_ret) << endl;
		cl_get_compiler_err_log(program, device);
		return 1;
	}
	return 0;
}

/**
* Read kernel from file
*/
void read_kernel(string file_name, string &str_kernel)
{
	ifstream in_file(file_name.c_str());
	in_file.open(file_name.c_str());
	DIE( !in_file.is_open(), "ERR OpenCL kernel file. Same directory as binary ?" );

	stringstream str_stream;
	str_stream << in_file.rdbuf();

	str_kernel = str_stream.str();
}

/**
 * OpenCL return error message, used by CL_ERR and CL_COMPILE_ERR
 */
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

/**
 * Check compiler return code, used by CL_COMPILE_ERR
 */
void cl_get_compiler_err_log(cl_program program, cl_device_id device)
{
	char* build_log;
	size_t log_size;

	/* first call to know the proper size */
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
						  0, NULL, &log_size);
	build_log = new char[ log_size + 1 ];

	/* second call to get the log */
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
						  log_size, build_log, NULL);
	build_log[ log_size ] = '\0';
	cout << endl << build_log << endl;
}
>>>> file: helper.hpp
#ifndef CL_HELPER_H
#define CL_HELPER_H

#if __APPLE__
   #include 
#else
   #include 
#endif

#include 

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
>>>> file: kernel.cl
#define BLOCK_ELEMENTS 16

typedef union {
	uint bits;
	struct {
		uchar b;
		uchar g;
		uchar r;
		uchar a;
	} channels ;
	uchar components[4];
} Color;

uchar round_to_5_bits(float val) {
	uchar v = (uchar)(val * 31.0f / 255.0f + 0.5f);
	uchar low = 0;
	uchar high = 31;
	return clamp(v, low, high);
}

uchar round_to_4_bits(float val) {
	uchar v = (uchar)(val * 15.0f / 255.0f + 0.5f);
	uchar low = 0;
	uchar high = 15;
	return clamp(v, low, high);
}

__constant short g_codeword_tables[8][4] __attribute__((aligned(16))) = {
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
	{0, 4, 1, 5, 2, 6, 3, 7},        // Vertical block 0.
	{8, 12, 9, 13, 10, 14, 11, 15},  // Vertical block 1.
	{0, 4, 8, 12, 1, 5, 9, 13},      // Horizontal block 0.
	{2, 6, 10, 14, 3, 7, 11, 15}     // Horizontal block 1.
};

__constant uchar two_compl_trans_table[8] = {
	4,  // -4 (100b)
	5,  // -3 (101b)
	6,  // -2 (110b)
	7,  // -1 (111b)
	0,  //  0 (000b)
	1,  //  1 (001b)
	2,  //  2 (010b)
	3,  //  3 (011b)
};

Color makeColor(Color base, short lum) {
	int b = (int)(base.channels.b) + lum;
	int g = (int)(base.channels.g) + lum;
	int r = (int)(base.channels.r) + lum;
	Color color;
	color.channels.b = (uchar)(clamp(b, 0, 255));
	color.channels.g = (uchar)(clamp(g, 0, 255));
	color.channels.r = (uchar)(clamp(r, 0, 255));
	return color;
}

uint getColorError(Color u, Color v) {
	int delta_b = (int)(u.channels.b) - v.channels.b;
	int delta_g = (int)(u.channels.g) - v.channels.g;


	int delta_r = (int)(u.channels.r) - v.channels.r;
	return delta_b * delta_b + delta_g * delta_g + delta_r * delta_r;
}

void WriteColors444(uchar* block,
					Color color0,
					Color color1) {
	block[0] = (color0.channels.r & 0xf0) | (color1.channels.r >> 4);
	block[1] = (color0.channels.g & 0xf0) | (color1.channels.g >> 4);


	block[2] = (color0.channels.b & 0xf0) | (color1.channels.b >> 4);
}

void WriteColors555(uchar* block,
					Color color0,
					Color color1) {
	short delta_r = (short)(color1.channels.r >> 3) - (color0.channels.r >> 3);
	short delta_g = (short)(color1.channels.g >> 3) - (color0.channels.g >> 3);
	short delta_b = (short)(color1.channels.b >> 3) - (color0.channels.b >> 3);
	
	block[0] = (color0.channels.r & 0xf8) | two_compl_trans_table[delta_r + 4];


	block[1] = (color0.channels.g & 0xf8) | two_compl_trans_table[delta_g + 4];
	block[2] = (color0.channels.b & 0xf8) | two_compl_trans_table[delta_b + 4];
}

void WriteCodewordTable(uchar* block,
							   uchar sub_block_id,
							   uchar table) {
	
	uchar shift = (2 + (3 - sub_block_id * 3));
	block[3] &= ~(0x07 << shift);
	block[3] |= table << shift;
}

void WritePixelData(uchar* block, uint pixel_data) {
	block[4] |= pixel_data >> 24;
	block[5] |= (pixel_data >> 16) & 0xff;


	block[6] |= (pixel_data >> 8) & 0xff;
	block[7] |= pixel_data & 0xff;
}

void WriteFlip(uchar* block, uchar flip) {
	block[3] &= ~0x01;
	block[3] |= flip;
}

void WriteDiff(uchar* block, uchar diff) {
	block[3] &= ~0x02;


	block[3] |= diff << 1;
}

Color makeColor444(const float* bgr) {
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

Color makeColor555(const float* bgr) {
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
	
void getAverageColor(__private const Color* src, float* avg_color)
{
	uint sum_b = 0, sum_g = 0, sum_r = 0;
	
	for (int i = 0; i < 8; ++i) {
		sum_b += src[i].channels.b;
		sum_g += src[i].channels.g;
		sum_r += src[i].channels.r;
	}
	
	const float kInv8 = 1.0f / 8.0f;
	avg_color[0] = (float)(sum_b) * kInv8;
	avg_color[1] = (float)(sum_g) * kInv8;
	avg_color[2] = (float)(sum_r) * kInv8;
}


int SolveSolidBlock(Color vert[BLOCK_ELEMENTS],
					uchar* block) {
	for (int i = 1; i < BLOCK_ELEMENTS; i++) {
		if (vert[i].bits != vert[0].bits)
			return 0;
	}

	float src_color_float[3] = {
		(float)(vert[0].channels.b),
		(float)(vert[0].channels.g),
		(float)(vert[0].channels.r)
	};
	Color base = makeColor555(src_color_float);

	WriteDiff(block, 1);
	WriteFlip(block, 0);
	WriteColors555(block, base, base);

	uchar best_tbl_idx = 0;
	uchar best_mod_idx = 0;
	uint best_mod_err = UINT_MAX;
	
	for (uint tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		for (uint mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			Color color = makeColor(base, lum);

			uint mod_err = getColorError(vert[0], color);
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
	
	WriteCodewordTable(block, 0, best_tbl_idx);
	WriteCodewordTable(block, 1, best_tbl_idx);
	
	uchar pix_idx = g_mod_to_pix[best_mod_idx];
	uint lsb = pix_idx & 0x1;
	uint msb = pix_idx >> 1;
	
	uint pix_data = 0;
	for (uint i = 0; i < 2; ++i) {
		for (uint j = 0; j < 8; ++j) {
			int texel_num = g_idx_to_num[i][j];
			pix_data |= msb << (texel_num + 16);
			pix_data |= lsb << (texel_num);
		}
	}
	
	WritePixelData(block, pix_data);
}

void computeLuminance(uchar* block,
					  __private const Color* src,
					  Color base,
					  int sub_block_id,
					  __constant const uchar* idx_to_num_tab)
{
	uint best_tbl_err = INT_MAX;
	uchar best_tbl_idx = 0;
	uchar best_mod_idx[8][8];

	for (int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		Color candidate_color[4];
		for (int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			candidate_color[mod_idx] = makeColor(base, lum);
		}
		
		uint tbl_err = 0;
		
		for (int i = 0; i < 8; ++i) {
			uint best_mod_err = INT_MAX;
			for (int mod_idx = 0; mod_idx < 4; ++mod_idx) {
				Color color = candidate_color[mod_idx];

				uint mod_err = getColorError(src[i], color);
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

	for (int i = 0; i < 8; ++i) {
		uchar mod_idx = best_mod_idx[best_tbl_idx][i];
		uchar pix_idx = g_mod_to_pix[mod_idx];
		
		uint lsb = pix_idx & 0x1;
		uint msb = pix_idx >> 1;
		
		// Obtain the texel number as specified in the standard.
		int texel_num = idx_to_num_tab[i];
		pix_data |= msb << (texel_num + 16);
		pix_data |= lsb << (texel_num);
	}

	WritePixelData(block, pix_data);
}

void compress(__private const Color* vert,
			  __private const Color* horz,
			  uchar* block) {
	if (SolveSolidBlock(vert, block)) {
		return;
	}

	__private const Color* sub_block_src[4] = {
		vert, vert + 8,
		horz, horz + 8
	};
	Color sub_block_avg[4];
	int use_differential[2] = {1, 1};

	for (uint i = 0, j = 1; i < 4; i += 2, j += 2) {
		float avg_color_0[3];
		getAverageColor(sub_block_src[i], avg_color_0);
		Color avg_color_555_0 = makeColor555(avg_color_0);

		float avg_color_1[3];
		getAverageColor(sub_block_src[j], avg_color_1);
		Color avg_color_555_1 = makeColor555(avg_color_1);
		
		for (int light_idx = 0; light_idx < 3; ++light_idx) {
			int u = avg_color_555_0.components[light_idx] >> 3;
			int v = avg_color_555_1.components[light_idx] >> 3;
			
			int component_diff = v - u;
			if (component_diff  3) {
				use_differential[i / 2] = 0;
				sub_block_avg[i] = makeColor444(avg_color_0);
				sub_block_avg[j] = makeColor444(avg_color_1);
			} else {
				sub_block_avg[i] = avg_color_555_0;
				sub_block_avg[j] = avg_color_555_1;
			}
		}
	}

	uint sub_block_err[4] = {0, 0, 0, 0};
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 8; ++j) {
			sub_block_err[i] += getColorError(sub_block_avg[i], sub_block_src[i][j]);
		}
	}
	
	int flip =
	sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1];
	
	WriteDiff(block, use_differential[!!flip]);
	WriteFlip(block, flip);
	
	uchar sub_block_off_0 = flip ? 2 : 0;
	uchar sub_block_off_1 = sub_block_off_0 + 1;
	
	if (use_differential[!!flip]) {
		WriteColors555(block, sub_block_avg[sub_block_off_0],
					   sub_block_avg[sub_block_off_1]);
	} else {
		WriteColors444(block, sub_block_avg[sub_block_off_0],
					   sub_block_avg[sub_block_off_1]);
	}
	
	computeLuminance(block, sub_block_src[sub_block_off_0],
								   sub_block_avg[sub_block_off_0], 0,
								   g_idx_to_num[sub_block_off_0]);
	computeLuminance(block, sub_block_src[sub_block_off_1],
								   sub_block_avg[sub_block_off_1], 1,
								   g_idx_to_num[sub_block_off_1]);
}

__kernel void
kernel_main(__global const Color* const src,
			__global uchar* restrict dst,
			uint width,
			uint height) {

	Color horz[BLOCK_ELEMENTS];
	Color vert[BLOCK_ELEMENTS];

	int gi = get_global_id(0);
	int gj = get_global_id(1);

	__global const Color* const row0 = src + gi * 4 * width + gj * 4;
	__global const Color* const row1 = row0 + width;
	__global const Color* const row2 = row1 + width;
	__global const Color* const row3 = row2 + width;

	vert[0].bits = row0[0].bits;
	vert[1].bits = row0[1].bits;
	vert[2].bits = row1[0].bits;
	vert[3].bits = row1[1].bits;
	vert[4].bits = row2[0].bits;
	vert[5].bits = row2[1].bits;
	vert[6].bits = row3[0].bits;
	vert[7].bits = row3[1].bits;
	vert[8].bits = row0[2].bits;
	vert[9].bits = row0[3].bits;
	vert[10].bits = row1[2].bits;
	vert[11].bits = row1[3].bits;
	vert[12].bits = row2[2].bits;
	vert[13].bits = row2[3].bits;
	vert[14].bits = row3[2].bits;
	vert[15].bits = row3[3].bits;

	horz[0].bits = row0[0].bits;
	horz[1].bits = row0[1].bits;
	horz[2].bits = row0[2].bits;
	horz[3].bits = row0[3].bits;
	horz[4].bits = row1[0].bits;
	horz[5].bits = row1[1].bits;
	horz[6].bits = row1[2].bits;
	horz[7].bits = row1[3].bits;
	horz[8].bits = row2[0].bits;
	horz[9].bits = row2[1].bits;
	horz[10].bits = row2[2].bits;
	horz[11].bits = row2[3].bits;
	horz[12].bits = row3[0].bits;
	horz[13].bits = row3[1].bits;
	horz[14].bits = row3[2].bits;
	horz[15].bits = row3[3].bits;

	uchar block[8];
	for (int i = 0; i < 8; i++)
		 block[i] = 0;
	compress(vert, horz, block);

	for (int i = 0; i < 8; i++)
		dst[(gi * (width / 4) + gj) * 8 + i] = block[i];
}
>>>> file: texture_compress_skl.cpp
#include "compress.hpp"
#include "helper.hpp"

#include 
#include 
#include 
#include 

#include 
#include 

using namespace std;

namespace {

cl_device_id FindGPU(cl_device_id* &device_ids, cl_platform_id* &platform_ids) {
	cl_uint num_platforms;
	cl_uint num_devices;
	cl_device_id device;

	CL_ERR( clGetPlatformIDs(0, NULL, &num_platforms) );
	platform_ids = new cl_platform_id[num_platforms];


	DIE(platform_ids == NULL, "alloc platform_ids");

	CL_ERR( clGetPlatformIDs(num_platforms, platform_ids, NULL) );

	for (uint i = 0; i < num_platforms; i++) {
		cl_platform_id platform = platform_ids[i];
		DIE(platform == 0, "platform selection");

		if (clGetDeviceIDs(platform,
			CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices) == CL_DEVICE_NOT_FOUND) {
			num_devices = 0;
			continue;
		}



		device_ids = new cl_device_id[num_devices];
		DIE(device_ids == NULL, "alloc devicei_ids");

		CL_ERR( clGetDeviceIDs(platform,
			CL_DEVICE_TYPE_GPU, num_devices, device_ids, NULL) );

		device = device_ids[0];
		break;
	}

	return device;
}

const char kKernelPath[] = "kernel.cl";
const char kKernelFn[] = "kernel_main";
const int kCompressionRatio = 8;

} // anonymous namespace

TextureCompressor::TextureCompressor() {
	cl_int ret;

	device = FindGPU(device_ids, platform_ids);

	context = clCreateContext(0, 1, &device, NULL, NULL, &ret);
	CL_ERR( ret );

	{
		string kernel_str;
		read_kernel(kKernelPath, kernel_str);
		const char* kernel_str_c = kernel_str.c_str();

		program = clCreateProgramWithSource(context,
			1, &kernel_str_c, NULL, &ret);
		CL_ERR( ret );

		ret = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
		CL_COMPILE_ERR( ret, program, device );

		kernel = clCreateKernel(program, kKernelFn, &ret);
		CL_ERR( ret );
	}
}

TextureCompressor::~TextureCompressor() {
	CL_ERR( clReleaseContext(context) );

	delete[] device_ids;
	delete[] platform_ids;
}
	
unsigned long TextureCompressor::compress(const uint8_t* src,
									  uint8_t* dst,
									  int width,
									  int height)
{
	cl_int ret;
	cl_mem src_device, dst_device;

	command_queue = clCreateCommandQueue(context, device, 0, &ret);
	CL_ERR( ret );

	src_device = clCreateBuffer(context, CL_MEM_READ_WRITE,
		sizeof(cl_uint) * width * height, NULL, &ret);
	CL_ERR( ret );

	dst_device = clCreateBuffer(context, CL_MEM_READ_WRITE,
		sizeof(cl_uint) * width * height / kCompressionRatio, NULL, &ret);
	CL_ERR( ret );

	CL_ERR( clEnqueueWriteBuffer(command_queue, src_device, CL_TRUE, 0,
		sizeof(cl_uint) * width * height, src, 0, NULL, NULL) );

	CL_ERR( clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&src_device) );
	CL_ERR( clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&dst_device) );
	CL_ERR( clSetKernelArg(kernel, 2, sizeof(cl_uint), (void*)&width) );
	CL_ERR( clSetKernelArg(kernel, 3, sizeof(cl_uint), (void*)&height) );

	size_t globalWorkSize[2] = {height / 4, width / 4};
	size_t localWorkSize[2] = {1, 1};
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL,
		globalWorkSize, NULL, 0, NULL, NULL);
	CL_ERR( ret );

	CL_ERR( clEnqueueReadBuffer(command_queue, dst_device, CL_TRUE, 0,
		sizeof(cl_uint) * width * height / kCompressionRatio, dst, 0, NULL,
		NULL) );

	CL_ERR( clFinish(command_queue) );

	CL_ERR( clReleaseMemObject(src_device) );
	CL_ERR( clReleaseMemObject(dst_device) );

	CL_ERR( clReleaseCommandQueue(command_queue) );

	return 0;
}
