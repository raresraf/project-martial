

typedef uchar uint8_t;

#define ALIGNAS(X)	__attribute__((aligned(X)))

union Color {
	struct BgraColorType {
		uint8_t b;
		uint8_t g;
		uint8_t r;
		uint8_t a;
	} channels;
	uint8_t components[4];
	uint bits;
};

inline uint8_t myclamp(uint8_t val, uint8_t min, uint8_t max) {
	return val  max ? max : val);
}

inline uint8_t round_to_5_bits(float val) {
	return myclamp(val * 31.0f / 255.0f + 0.5f, 0, 31);
}

inline uint8_t round_to_4_bits(float val) {
	return myclamp(val * 15.0f / 255.0f + 0.5f, 0, 15);
}



ALIGNAS(16) __constant static const int g_codeword_tables[8][4] = {
	{-8, -2, 2, 8},
	{-17, -5, 5, 17},
	{-29, -9, 9, 29},
	{-42, -13, 13, 42},
	{-60, -18, 18, 60},
	{-80, -24, 24, 80},
	{-106, -33, 33, 106},
	{-183, -47, 47, 183}};



__constant static const uint8_t g_mod_to_pix[4] = {3, 2, 0, 1};





















__constant static const uint8_t g_idx_to_num[4][8] = {
	{0, 4, 1, 5, 2, 6, 3, 7},        
	{8, 12, 9, 13, 10, 14, 11, 15},  
	{0, 4, 8, 12, 1, 5, 9, 13},      
	{2, 6, 10, 14, 3, 7, 11, 15}     
};


inline union Color makeColor(const union Color base, int lum) {
	int b = (int)(base.channels.b) + lum;


	int g = (int)(base.channels.g) + lum;
	int r = (int)(base.channels.r) + lum;
	union Color color;
	color.channels.b = (uint8_t)(clamp(b, 0, 255));
	color.channels.g = (uint8_t)(clamp(g, 0, 255));
	color.channels.r = (uint8_t)(clamp(r, 0, 255));
	return color;
}



inline uint getColorError(const union Color u, const union Color v) {
#ifdef USE_PERCEIVED_ERROR_METRIC
	float delta_b = (float)(u.channels.b) - v.channels.b;
	float delta_g = (float)(u.channels.g) - v.channels.g;
	float delta_r = (float)(u.channels.r) - v.channels.r;
	return (uint)(0.299f * delta_b * delta_b +
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
						   const union Color color0,
						   const union Color color1) {
	
	block[0] = (color0.channels.r & 0xf0) | (color1.channels.r >> 4);
	block[1] = (color0.channels.g & 0xf0) | (color1.channels.g >> 4);
	block[2] = (color0.channels.b & 0xf0) | (color1.channels.b >> 4);
}

inline void WriteColors555(__global uint8_t* block,
						   const union Color color0,
						   const union Color color1) {
	
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
	
	int delta_r =
	(int)(color1.channels.r >> 3) - (color0.channels.r >> 3);
	int delta_g =
	(int)(color1.channels.g >> 3) - (color0.channels.g >> 3);
	int delta_b =
	(int)(color1.channels.b >> 3) - (color0.channels.b >> 3);
	
	
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

inline void WritePixelData(__global uint8_t* block, uint pixel_data) {
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

inline void ExtractBlock(uint8_t* dst, const uint8_t* src, int width) {
	for (int j = 0; j < 4; ++j) {
		
		for (int i = 0; i < 16; i++) {
			dst[j * 4 * 4 + i] = *(src + i);
		}
		src += width * 4;
	}
}





inline union Color makeColor444(const float* bgr) {
	uint8_t b4 = round_to_4_bits(bgr[0]);
	uint8_t g4 = round_to_4_bits(bgr[1]);
	uint8_t r4 = round_to_4_bits(bgr[2]);
	union Color bgr444;
	bgr444.channels.b = (b4 << 4) | b4;
	bgr444.channels.g = (g4 << 4) | g4;
	bgr444.channels.r = (r4 << 4) | r4;
	
	bgr444.channels.a = 0x44;
	return bgr444;
}





inline union Color makeColor555(const float* bgr) {
	uint8_t b5 = round_to_5_bits(bgr[0]);
	uint8_t g5 = round_to_5_bits(bgr[1]);
	uint8_t r5 = round_to_5_bits(bgr[2]);
	union Color bgr555;
	bgr555.channels.b = (b5 > 2);
	bgr555.channels.g = (g5 > 2);
	bgr555.channels.r = (r5 > 2);
	
	bgr555.channels.a = 0x55;
	return bgr555;
}
	
void getAverageColor(const union Color* src, float* avg_color)
{
	uint sum_b = 0, sum_g = 0, sum_r = 0;
	
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
						   const union Color* src,
						   const union Color base,
						   int sub_block_id,
						   __constant const uint8_t* idx_to_num_tab,
						   unsigned long threshold)
{
	uint best_tbl_err = threshold;
	uint8_t best_tbl_idx = 0;
	uint8_t best_mod_idx[8][8];  

	
	
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		
		
		union Color candidate_color[4];  
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			int lum = g_codeword_tables[tbl_idx][mod_idx];
			candidate_color[mod_idx] = makeColor(base, lum);
		}
		
		uint tbl_err = 0;
		


		for (unsigned int i = 0; i < 8; ++i) {
			
			
			uint best_mod_err = threshold;
			for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
				const union Color color = candidate_color[mod_idx];
				
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

	for (unsigned int i = 0; i < 8; ++i) {
		uint8_t mod_idx = best_mod_idx[best_tbl_idx][i];
		uint8_t pix_idx = g_mod_to_pix[mod_idx];
		
		uint lsb = pix_idx & 0x1;
		uint msb = pix_idx >> 1;
		
		
		int texel_num = idx_to_num_tab[i];
		pix_data |= msb << (texel_num + 16);


		pix_data |= lsb << (texel_num);
	}

	WritePixelData(block, pix_data);

	return best_tbl_err;
}


bool tryCompressSolidBlock(__global uint8_t* dst,
						   const union Color* src,
						   unsigned long* error)
{
	for (unsigned int i = 1; i < 16; ++i) {
		if (src[i].bits != src[0].bits)
			return false;
	}
	
	
	
	__global uint8_t * it = dst;
	for (int i = 0; i < 8; i++) {
		*it = 0;
	}
	
	float src_color_float[3] = {(float)(src->channels.b),
		(float)(src->channels.g),
		(float)(src->channels.r)};
	union Color base = makeColor555(src_color_float);
	
	WriteDiff(dst, true);
	WriteFlip(dst, false);
	WriteColors555(dst, base, base);
	
	uint8_t best_tbl_idx = 0;
	uint8_t best_mod_idx = 0;
	uint best_mod_err = UINT_MAX; 
	
	
	
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		
		
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {


			int lum = g_codeword_tables[tbl_idx][mod_idx];
			const union Color color = makeColor(base, lum);
			
			uint mod_err = getColorError(*src, color);
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
	return true;
}

unsigned long compressBlock(__global uint8_t* dst,
						   const union Color* ver_src,
						   const union Color* hor_src,
						   unsigned long threshold)
{
	unsigned long solid_error = 0;
	if (tryCompressSolidBlock(dst, ver_src, &solid_error)) {		
		return solid_error;
	}
	
	const union Color* sub_block_src[4] = {ver_src, ver_src + 8, hor_src, hor_src + 8};
	
	union Color sub_block_avg[4];
	bool use_differential[2] = {true, true};
	
	
	
	for (unsigned int i = 0, j = 1; i < 4; i += 2, j += 2) {
		float avg_color_0[3];
		getAverageColor(sub_block_src[i], avg_color_0);
		union Color avg_color_555_0 = makeColor555(avg_color_0);
		
		float avg_color_1[3];
		getAverageColor(sub_block_src[j], avg_color_1);
		union Color avg_color_555_1 = makeColor555(avg_color_1);
		
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
	
	
	
	
	uint sub_block_err[4] = {0};
	for (unsigned int i = 0; i < 4; ++i) {
		for (unsigned int j = 0; j < 8; ++j) {
			sub_block_err[i] += getColorError(sub_block_avg[i], sub_block_src[i][j]);
		}
	}
	
	bool flip =
	sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1];
	
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

void color_memcpy(union Color * dst, __global const union Color * src, int numBytes) {
	uint8_t *it_dst = (uint8_t *)dst;
	__global uint8_t *it_src = (__global uint8_t *)src;
	for (int i = 0; i < numBytes; i++) {
		it_dst[i] = it_src[i];
	}
}


__kernel void kernel_computation(int width, int height,
				__global uint8_t *src, __global uint8_t *dst)
{
	int l = get_global_id(0); 
	int c = get_global_id(1); 

	
	union Color ver_blocks[16];
	union Color hor_blocks[16];	

	
	
	if (l * 4 + 4 > height) {
		return;	
	}

	
	
	src += (l * width * 4 * 4 + c * 4 * 4);
	dst += (l * width * 2 + c * 8);

	
	__global const union Color* row0 = (__global const union Color *)(src);
	__global const union Color* row1 = row0 + width;
	__global const union Color* row2 = row1 + width;
	__global const union Color* row3 = row2 + width;
	
	color_memcpy(ver_blocks, row0, 8);
	color_memcpy(ver_blocks + 2, row1, 8);
	color_memcpy(ver_blocks + 4, row2, 8);
	color_memcpy(ver_blocks + 6, row3, 8);
	color_memcpy(ver_blocks + 8, row0 + 2, 8);
	color_memcpy(ver_blocks + 10, row1 + 2, 8);
	color_memcpy(ver_blocks + 12, row2 + 2, 8);
	color_memcpy(ver_blocks + 14, row3 + 2, 8);
	
	color_memcpy(hor_blocks, row0, 16);
	color_memcpy(hor_blocks + 4, row1, 16);
	color_memcpy(hor_blocks + 8, row2, 16);
	color_memcpy(hor_blocks + 12, row3, 16);

	
	for (int i = 0; i < 8; i++) {
		dst[i] = 0;
	}
	
	
	compressBlock(dst, ver_blocks, hor_blocks, UINT_MAX);
}>>>> file: texture_compress_skl.cpp
#include "compress.hpp"

#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include 

using namespace std;


#define DEVICE_SELECTION 0


void printError(const char* errMsg) {
	fprintf(stderr, errMsg);
	exit(-1);
}


void gpu_find(cl_device_id &device) {

	
	cl_platform_id platform;
	cl_uint platform_num = 0;
	cl_platform_id* platform_list = NULL;

	
	cl_uint device_num = 0;
	cl_device_id* device_list = NULL;

	
	if (clGetPlatformIDs(0, NULL, &platform_num) == CL_INVALID_VALUE) {
		printError("clGetPlatformIDs problems!\n");
	}

	
	if (platform_num <= 0) {
		printError("There are no available platforms!\n");
	}

	
	platform_list = new cl_platform_id[platform_num];
	if (!platform_list) {
		printError("The list of platforms was not properly allocated!\n");
	}

	
	if (clGetPlatformIDs(platform_num, platform_list, NULL) == CL_INVALID_VALUE) {
		printError("clGetPlatformIDs problems!\n");
	}

	
	for(uint platf = 0; platf < platform_num; platf++)
	{
		
		platform = platform_list[platf];
		if (!platform) {
			printError("Invalid platform selection!\n");
		}

		
		if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU,
			0, NULL, &device_num) == CL_DEVICE_NOT_FOUND) {
			device_num = 0;

			
			continue;
		}

		
		device_list = new cl_device_id[device_num];
		if (!device_list) {
			printError("The list of devices was not properly allocated!\n");
		}

		
		if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU,
			device_num, device_list, NULL) == CL_INVALID_VALUE) {
			printError("clGetDeviceIDs problems!\n");
		}
	
		
		device = device_list[DEVICE_SELECTION];
		
		
		delete[] device_list;

		
		break;
	}

	
	delete[] platform_list;	
}


void read_kernel(string file_name, string &str_kernel)
{
	
	ifstream in_file(file_name.c_str());
	if (!in_file.is_open()) {
		printError("Problems with file opening process!\n");
	}

	
	stringstream str_stream;
	str_stream << in_file.rdbuf();
	str_kernel = str_stream.str();
}


void gpu_execute_kernel(cl_device_id device, size_t dim1, size_t dim2,
						size_t width, size_t height,
						const uint8_t *src, uint8_t *dst)
{
	
	cl_context context;

	
	cl_command_queue cmd_queue;

	
	string kernel_src;

	
	cl_program program;

	
	cl_kernel kernel;

	
	cl_mem src_dev;
	cl_mem dst_dev;

	
	cl_int ret;

	
	context = clCreateContext(0, 1, &device, NULL, NULL, &ret);
	if (ret == CL_INVALID_VALUE) {
		printError("clCreateContext problems!\n");
	}

	
	cmd_queue = clCreateCommandQueue(context, device, 0, &ret);
	if (ret == CL_INVALID_VALUE) {
		printError("clCreateCommandQueue problems!\n");
	}

	
	src_dev = clCreateBuffer(context, CL_MEM_READ_ONLY,
				  sizeof(uint8_t) * width * height * 4, NULL, &ret);
	if (ret == CL_INVALID_VALUE) {
		printError("clCreateBuffer problems!\n");
	}

	
	dst_dev = clCreateBuffer(context, CL_MEM_READ_WRITE,
				  sizeof(uint8_t) * width * height * 4 / 8, NULL, &ret);
	if (ret == CL_INVALID_VALUE) {
		printError("clCreateBuffer problems!\n");
	}

	
	if (clEnqueueWriteBuffer(cmd_queue, src_dev, CL_TRUE, 0, 
		sizeof(uint8_t) * width * height * 4,
		src, 0, NULL, NULL) == CL_INVALID_VALUE) {
		printError("clEnqueueWriteBuffer problems!\n");
	}

	
	read_kernel("skl_device.cl", kernel_src);
	const char* kernel_c_str = kernel_src.c_str();

	
	program = clCreateProgramWithSource(context, 1,
		  &kernel_c_str, NULL, &ret);
	if (ret == CL_INVALID_VALUE) {
		printError("clCreateProgramWithSource problems!\n");
	}

	
	ret = clBuildProgram(program, 1, &device, "-Werror", NULL, NULL);
	if(ret != CL_SUCCESS) {
		printError("clBuildProgram problems!\n");
	}

	
	kernel = clCreateKernel(program, "kernel_computation", &ret);
	if (ret == CL_INVALID_VALUE) {
		printError("clCreateKernel problems!\n");
	}

	
	if (clSetKernelArg(kernel, 0, sizeof(cl_uint),
		(void *)&width) == CL_INVALID_VALUE) {
		printError("clSetKernelArg problems 0!\n");
	}
	if (clSetKernelArg(kernel, 1, sizeof(cl_uint),
		(void *)&height) == CL_INVALID_VALUE) {
		printError("clSetKernelArg problems 1!\n");
	}
	if (clSetKernelArg(kernel, 2, sizeof(cl_mem),
		(void *)&src_dev) == CL_INVALID_VALUE) {
		printError("clSetKernelArg problems 2!\n");
	}
	if (clSetKernelArg(kernel, 3, sizeof(cl_mem),
		(void *)&dst_dev) == CL_INVALID_VALUE) {
		printError("clSetKernelArg problems 3!\n");
	}

	
	size_t globalSize[2] = {height / 4, width /4};

	
	if (clEnqueueNDRangeKernel(cmd_queue, kernel, 2, NULL, globalSize,
		NULL, 0,  NULL, NULL) != CL_SUCCESS) {
		printError("clEnqueueNDRangeKernel problems!\n\n");
	}

	
	if (clEnqueueReadBuffer(cmd_queue, dst_dev, CL_TRUE, 0,
		  sizeof(uint8_t) * width * height * 4 / 8,
		  dst, 0, NULL, NULL) == CL_INVALID_VALUE) {
		printError("clEnqueueReadBuffer problems!\n");
	}

	
	if (clFinish(cmd_queue) == CL_INVALID_VALUE) {
		printError("clFinish problems!\n");
	}

	
	if (clReleaseMemObject(src_dev) == CL_INVALID_VALUE) {
		printError("clReleaseMemObject problems!\n");
	}
	if (clReleaseMemObject(dst_dev) == CL_INVALID_VALUE) {
		printError("clReleaseMemObject problems!\n");
	}
	if (clReleaseCommandQueue(cmd_queue) == CL_INVALID_VALUE) {
		printError("clReleaseCommandQueue problems!\n");
	}
	if (clReleaseContext(context) == CL_INVALID_VALUE) {
		printError("clReleaseContext problems!\n");
	}
}

TextureCompressor::TextureCompressor() {
	
	gpu_find(device);
}

TextureCompressor::~TextureCompressor() { }
	
unsigned long TextureCompressor::compress(const uint8_t* src, uint8_t* dst,
									  int width, int height)
{
	
	gpu_execute_kernel(device, 4, 4, (size_t)width, (size_t)height, src, dst);

	
	srand(time(NULL));
	return rand();
}
