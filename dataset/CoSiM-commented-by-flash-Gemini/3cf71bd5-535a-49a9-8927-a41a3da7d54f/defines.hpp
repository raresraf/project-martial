
#ifndef DEFINES_H
#define DEFINES_H

#define SELECT_VENDOR "NVIDIA"
#define DEVICE_NO 1
#define KERNEL_FILE "kernel.cl"
#define KERNEL_FUNCTION "start"

#endif
#include 
#include 
#include 
#include 
#include 

#include "helper.hpp"

using namespace std;


void CL_ERR(int cl_ret)
{
  DIE(cl_ret != CL_SUCCESS, cl_get_string_err(cl_ret));
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

#include 

using namespace std;

void CL_ERR(int cl_ret);
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
#define INT32_MAX 0xffffffff

typedef union Color {
	struct BgraColorType {
		uchar b;
		uchar g;
		uchar r;
		uchar a;
	} channels;
	uchar components[4];
	uint bits;
} Color;



__constant short g_codeword_tables[8][4] = {
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

void memset_global(__global uchar *dst, uchar value, size_t size) {
	size_t i;
	for (i = 0; i < size; ++i) {
		dst[i] = value;
	}
}

void memcpy_global_private(uchar *dst, __global uchar *src, size_t size) {
	size_t i;
	for (i = 0; i < size; ++i) {
		dst[i] = src[i];
	}
}

uchar clamp_int_to_uchar(int val, uchar min, uchar max) {
	return val  max ? max : val);
}

uchar round_to_5_bits(float val) {
	return clamp_int_to_uchar(val * 31.0f / 255.0f + 0.5f, 0, 31);
}

uchar round_to_4_bits(float val) {
	return clamp_int_to_uchar(val * 15.0f / 255.0f + 0.5f, 0, 15);
}

Color makeColor(Color base, int lum);
uint getColorError(Color u, Color v);

void WriteColors444(__global uchar *block, Color color0, Color color1);
void WriteColors555(__global uchar *block, Color color0, Color color1);
void WriteCodewordTable(__global uchar *block, uchar sub_block_id, uchar table);
void WritePixelData(__global uchar *block, uint pixel_data);
void WriteFlip(__global uchar *block, uchar flip);
void WriteDiff(__global uchar *block, uchar diff);

Color makeColor444(float *bgr);
Color makeColor555(float *bgr);
void getAverageColor(Color *src, float *avg_color);

uint computeLuminance(__global uchar *block,
                    Color *src,
                    Color base,
                    int sub_block_id,
                    __constant uchar *idx_to_num_tab,
                    uint threshold);

uint compressBlock(__global uchar *dst,
			    Color *ver_src,
                Color *hor_src,
			    uint threshold);

__kernel void start(const __global uchar *src,
                    __global uchar *dst,
					__global uint *total_err,
                    const int width,
                    const int height)
{
    int block_row = get_global_id(0);
    int block_col = get_global_id(1);
    int block_src = block_row * width * 4 * 4 + block_col * 4 * 4;
    int block_dst = block_row * get_global_size(1) * 8 + block_col * 8;

    Color ver_blocks[16];
	Color hor_blocks[16];

    const __global Color *row0 = (const __global Color*)(src + block_src);
    const __global Color *row1 = row0 + width;
    const __global Color *row2 = row1 + width;
    const __global Color *row3 = row2 + width;
    
	memcpy_global_private(ver_blocks, row0, 8);
	memcpy_global_private(ver_blocks + 2, row1, 8);
	memcpy_global_private(ver_blocks + 4, row2, 8);
	memcpy_global_private(ver_blocks + 6, row3, 8);

	memcpy_global_private(ver_blocks + 8, row0 + 2, 8);
	memcpy_global_private(ver_blocks + 10, row1 + 2, 8);
	memcpy_global_private(ver_blocks + 12, row2 + 2, 8);
	memcpy_global_private(ver_blocks + 14, row3 + 2, 8);
    
	memcpy_global_private(hor_blocks, row0, 16);
	memcpy_global_private(hor_blocks + 4, row1, 16);
	memcpy_global_private(hor_blocks + 8, row2, 16);
	memcpy_global_private(hor_blocks + 12, row3, 16);

	memset_global(dst + block_dst, 0, 8);

	uint block_err;
	block_err = compressBlock(dst + block_dst, ver_blocks, hor_blocks, INT32_MAX);
	atomic_add(total_err, block_err);
}

uint compressBlock(__global uchar *dst,
			    Color *ver_src,
                Color *hor_src,
			    uint threshold) {

    uint solid_error = 0;
	Color* sub_block_src[4] = {ver_src, ver_src + 8, hor_src, hor_src + 8};
	
	Color sub_block_avg[4];
	uchar use_differential[2] = {1, 1};

    int i, j;

	
	
	for (i = 0, j = 1; i < 4; i += 2, j += 2) {
		float avg_color_0[3];
		getAverageColor(sub_block_src[i], avg_color_0);
		Color avg_color_555_0 = makeColor555(avg_color_0);
		
		float avg_color_1[3];
		getAverageColor(sub_block_src[j], avg_color_1);
		Color avg_color_555_1 = makeColor555(avg_color_1);

        uint light_idx;
		
		for (light_idx = 0; light_idx < 3; ++light_idx) {
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
	
	
	
	
	uint sub_block_err[4] = {0};
	for (i = 0; i < 4; ++i) {
		for (j = 0; j < 8; ++j) {
			sub_block_err[i] += getColorError(sub_block_avg[i], sub_block_src[i][j]);
		}
	}
	
    uchar flip;
    if (sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1]) {
        flip = 1;
    } else {
        flip = 0;
    }
	
	WriteDiff(dst, use_differential[flip]);
	WriteFlip(dst, flip);
	
	uchar sub_block_off_0 = flip ? 2 : 0;
	uchar sub_block_off_1 = sub_block_off_0 + 1;
	
	if (use_differential[flip]) {
		WriteColors555(dst, sub_block_avg[sub_block_off_0],
				   sub_block_avg[sub_block_off_1]);
	} else {
		WriteColors444(dst, sub_block_avg[sub_block_off_0],
				   sub_block_avg[sub_block_off_1]);
	}
	
	uint lumi_error1 = 0, lumi_error2 = 0;

	
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

uint computeLuminance(__global uchar *block,
                    Color *src,
                    Color base,
                    int sub_block_id,
                    __constant uchar *idx_to_num_tab,
                    uint threshold)
{
	uint best_tbl_err = threshold;
	uchar best_tbl_idx = 0;
	uchar best_mod_idx[8][8];  

    int i;
    int tbl_idx, mod_idx;

	
	
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


Color makeColor(const Color base, int lum) {
	int b = (int)(base.channels.b) + lum;
	int g = (int)(base.channels.g) + lum;
	int r = (int)(base.channels.r) + lum;
	Color color;
	color.channels.b = clamp_int_to_uchar(b, 0, 255);
	color.channels.g = clamp_int_to_uchar(g, 0, 255);


	color.channels.r = clamp_int_to_uchar(r, 0, 255);
	return color;
}



uint getColorError(Color u, Color v) {
	int delta_b = (int)(u.channels.b) - v.channels.b;
	int delta_g = (int)(u.channels.g) - v.channels.g;
	int delta_r = (int)(u.channels.r) - v.channels.r;
	return delta_b * delta_b + delta_g * delta_g + delta_r * delta_r;
}



void WriteColors444(__global uchar *block, Color color0, Color color1) {
	
	block[0] = (color0.channels.r & 0xf0) | (color1.channels.r >> 4);
	block[1] = (color0.channels.g & 0xf0) | (color1.channels.g >> 4);
	block[2] = (color0.channels.b & 0xf0) | (color1.channels.b >> 4);
}

void WriteColors555(__global uchar *block, Color color0, Color color1) {
	
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
	(short)(color1.channels.r >> 3) - (color0.channels.r >> 3);
	short delta_g =
	(short)(color1.channels.g >> 3) - (color0.channels.g >> 3);
	short delta_b =
	(short)(color1.channels.b >> 3) - (color0.channels.b >> 3);
	
	
	block[0] = (color0.channels.r & 0xf8) | two_compl_trans_table[delta_r + 4];
	block[1] = (color0.channels.g & 0xf8) | two_compl_trans_table[delta_g + 4];
	block[2] = (color0.channels.b & 0xf8) | two_compl_trans_table[delta_b + 4];
}

void WriteCodewordTable(__global uchar *block, uchar sub_block_id, uchar table) {
	
	uchar shift = (2 + (3 - sub_block_id * 3));
	block[3] &= ~(0x07 << shift);
	block[3] |= table << shift;
}

void WritePixelData(__global uchar *block, uint pixel_data) {
	block[4] |= pixel_data >> 24;
	block[5] |= (pixel_data >> 16) & 0xff;
	block[6] |= (pixel_data >> 8) & 0xff;
	block[7] |= pixel_data & 0xff;
}

void WriteFlip(__global uchar *block, uchar flip) {
	block[3] &= ~0x01;
	block[3] |= flip;
}



void WriteDiff(__global uchar *block, uchar diff) {
	block[3] &= ~0x02;
	block[3] |= diff << 1;
}





Color makeColor444(float *bgr) {
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





Color makeColor555(float *bgr) {
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
	
void getAverageColor(Color *src, float *avg_color)
{


    int i;
	uint sum_b = 0, sum_g = 0, sum_r = 0;
	
	for (int i = 0; i < 8; ++i) {
		sum_b += src[i].channels.b;
		sum_g += src[i].channels.g;
		sum_r += src[i].channels.r;
	}
	
	float kInv8 = 1.0f / 8.0f;
	avg_color[0] = (float)(sum_b) * kInv8;
	avg_color[1] = (float)(sum_g) * kInv8;
	avg_color[2] = (float)(sum_r) * kInv8;
}
#include "helper.hpp"
#include "lab8.hpp"
#include "../defines.hpp"

#include 
#include 
#include 

using namespace std;

void lab8 :: gpu_device_find(cl_device_id &device,
                cl_platform_id* platform_list,
				cl_device_id* device_list,
				uint dev_no)
{
	cl_platform_id platform;
	cl_uint platform_num = 0;
	bool select_platf;	

	cl_uint device_num = 0;

	size_t attr_size = 0;
	cl_char* attr_data = NULL;

	
	CL_ERR( clGetPlatformIDs(0, NULL, &platform_num));
	platform_list = new cl_platform_id[platform_num];
	DIE(platform_list == NULL, "alloc platform_list");

	
	CL_ERR( clGetPlatformIDs(platform_num, platform_list, NULL));
	cout << "Platforms found: " << platform_num << endl;

	
	for(uint platf=0; platf<platform_num; platf++)
	{
		
		CL_ERR( clGetPlatformInfo(platform_list[platf],
				CL_PLATFORM_VENDOR, 0, NULL, &attr_size));
		attr_data = new cl_char[attr_size];
		DIE(attr_data == NULL, "alloc attr_data");

		
		CL_ERR( clGetPlatformInfo(platform_list[platf],
				CL_PLATFORM_VENDOR, attr_size, attr_data, NULL));
		cout << "Platform " << platf << " " << attr_data << " ";
		
		
		if (strstr((char*)attr_data, SELECT_VENDOR)) {
			select_platf = true;
		} else {
			select_platf = false;
		}

		delete[] attr_data;

		
		CL_ERR( clGetPlatformInfo(platform_list[platf],
				CL_PLATFORM_VERSION, 0, NULL, &attr_size));
		attr_data = new cl_char[attr_size];
		DIE(attr_data == NULL, "alloc attr_data");

		
		CL_ERR( clGetPlatformInfo(platform_list[platf],
				CL_PLATFORM_VERSION, attr_size, attr_data, NULL));
		cout << attr_data << endl;
		delete[] attr_data;

		
		platform = platform_list[platf];
		DIE(platform == 0, "platform selection");

		
		if(clGetDeviceIDs(platform, 
			CL_DEVICE_TYPE_ALL, 0, NULL, &device_num) == CL_DEVICE_NOT_FOUND) {
			device_num = 0;
			continue;
		}

		
		if (select_platf && device_num <= dev_no) {
			dev_no = 0;
		}

		device_list = new cl_device_id[device_num];
		DIE(device_list == NULL, "alloc devices");

		
		CL_ERR( clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL,
			  device_num, device_list, NULL));
		cout << "\tDevices found " << device_num  << endl;

		
		for(uint dev=0; dev<device_num; dev++)
		{
			
			CL_ERR( clGetDeviceInfo(device_list[dev], CL_DEVICE_NAME,
				0, NULL, &attr_size));
			attr_data = new cl_char[attr_size];
			DIE(attr_data == NULL, "alloc attr_data");

			
			CL_ERR( clGetDeviceInfo(device_list[dev], CL_DEVICE_NAME,
				attr_size, attr_data, NULL));
			cout << "\tDevice " << dev << " " << attr_data << " ";
			delete[] attr_data;

			
			CL_ERR( clGetDeviceInfo(device_list[dev], CL_DEVICE_VERSION,
				0, NULL, &attr_size));
			attr_data = new cl_char[attr_size];
			DIE(attr_data == NULL, "alloc attr_data");

			
			CL_ERR( clGetDeviceInfo(device_list[dev], CL_DEVICE_VERSION,
				attr_size, attr_data, NULL));
			cout << attr_data; 
			delete[] attr_data;

			
			if (select_platf && (dev == dev_no)) {
				device = device_list[dev];
				cout << " <--- SELECTED ";
			}

			cout << endl;
		}
	}
}

void lab8 :: gpu_create_context(cl_device_id device,
                        cl_context &context,
                        cl_command_queue &cmdQueue)
{
    cl_int ret;
	
	
	context = clCreateContext(0, 1, &device, NULL, NULL, &ret);
	CL_ERR( ret );

	
	cmdQueue = clCreateCommandQueue(context, device, 0, &ret);
	CL_ERR( ret );
}

void lab8 :: gpu_compile_kernel(cl_device_id device,
                        cl_context &context,
                        cl_program &program,
                        cl_kernel &kernel,
                        string kernel_file)
{
    cl_int ret;
    string kernel_src;

    


	read_kernel(kernel_file.c_str(), kernel_src);
	const char* kernel_c_str = kernel_src.c_str();
	
	
	program = clCreateProgramWithSource(context, 1, &kernel_c_str, NULL, &ret);
	CL_ERR( ret );
	
	
	ret = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	CL_COMPILE_ERR( ret, program, device );
	
	
	kernel = clCreateKernel(program, KERNEL_FUNCTION, &ret);
	CL_ERR( ret );
}

unsigned long lab8 :: gpu_start(cl_context context,
                        cl_command_queue cmdQueue,
                        cl_kernel &kernel,
                        const uint8_t *src,
                        uint8_t *dst,
                        int width,
                        int height)
{
	unsigned long total_err = 0;
    cl_int ret;
	
	
	cl_mem bufSrc = clCreateBuffer(context, 
		CL_MEM_READ_ONLY, sizeof(uint8_t) * width * height * 4, NULL, &ret);
	CL_ERR( ret );
	cl_mem bufDst = clCreateBuffer(context, 
		CL_MEM_READ_WRITE, sizeof(uint8_t) * width * height / 2, NULL, &ret);
	CL_ERR( ret );
	cl_mem bufErr = clCreateBuffer(context, 
		CL_MEM_READ_WRITE, sizeof(uint32_t), NULL, &ret);
	CL_ERR( ret );
	CL_ERR( clEnqueueWriteBuffer(cmdQueue, bufSrc, CL_TRUE, 0,
		  sizeof(uint8_t) * width * height * 4, src, 0, NULL, NULL));
	CL_ERR( clEnqueueWriteBuffer(cmdQueue, bufErr, CL_TRUE, 0,


		  sizeof(uint32_t), &total_err, 0, NULL, NULL));
	
	
	CL_ERR( clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &bufSrc) );
	CL_ERR( clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &bufDst) );
	CL_ERR( clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &bufErr) );
	CL_ERR( clSetKernelArg(kernel, 3, sizeof(cl_int), (void *) &width) );
	CL_ERR( clSetKernelArg(kernel, 4, sizeof(cl_int), (void *) &height) );

	size_t globalSize[2] = {(size_t)height / 4, (size_t)width / 4};
	ret = clEnqueueNDRangeKernel(cmdQueue, 
		kernel, 2, NULL, globalSize, 0, 0, NULL, NULL);
	CL_ERR( ret );
	
	CL_ERR( clEnqueueReadBuffer(cmdQueue, bufDst, CL_TRUE, 0,
		  sizeof(uint8_t) * width * height / 2, dst, 0, NULL, NULL));
	CL_ERR( clEnqueueReadBuffer(cmdQueue, bufErr, CL_TRUE, 0,
		  sizeof(uint32_t), &total_err, 0, NULL, NULL));
	
	
	CL_ERR( clFinish(cmdQueue) );

	
	CL_ERR( clReleaseMemObject(bufSrc) );
	CL_ERR( clReleaseMemObject(bufDst) );
	CL_ERR( clReleaseMemObject(bufErr) );

	return total_err;
}
#ifndef LAB8_H
#define LAB8_H

#if __APPLE__
#include 
#else
#include 
#endif

namespace lab8 {
	void gpu_device_find(cl_device_id &device,
				cl_platform_id *platform_list,
				cl_device_id *device_list,
				uint dev_no = 0);

    void gpu_create_context(cl_device_id device,
                    cl_context &context,
					cl_command_queue &cmdQueue);

    void gpu_compile_kernel(cl_device_id device,
                    cl_context &context,
					cl_program &program,
					cl_kernel &kernel,
                    std::string kernel_file);

	unsigned long gpu_start(cl_context context,
                    cl_command_queue cmdQueue,
                    cl_kernel &kernel,
				    const uint8_t *src,
                    uint8_t *dst,
                    int width,
                    int height);
}

#endif
#include "compress.hpp"
#include "lab8/helper.hpp"
#include "lab8/lab8.hpp"
#include "defines.hpp"

#include 

using namespace std;

TextureCompressor::TextureCompressor() {
	lab8 :: gpu_device_find(this->device,
					this->platform_ids,
					this->device_ids,
					DEVICE_NO);
	lab8 :: gpu_create_context(this->device,
                    this->context,
					this->command_queue);
	lab8 :: gpu_compile_kernel(this->device,
                    this->context,
					this->program,
					this->kernel,
                    KERNEL_FILE);
}

TextureCompressor::~TextureCompressor() { 
	delete[] this->platform_ids;
	delete[] this->device_ids;

	CL_ERR(clReleaseCommandQueue(this->command_queue));
	CL_ERR(clReleaseContext(this->context));
}
	
unsigned long TextureCompressor::compress(const uint8_t* src,
								uint8_t* dst,
								int width,
								int height)
{
	return lab8 :: gpu_start(this->context, this->command_queue,
							this->kernel, src, dst, width, height);
}

unsigned long TextureCompressor::compressBlock(uint8_t* dst,
								const Color* ver_src,
								const Color* hor_src,
								unsigned long threshold)
{
	
	return INT32_MAX;
}
