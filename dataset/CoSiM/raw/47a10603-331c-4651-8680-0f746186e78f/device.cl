


union Color {
	struct BgraColorType {
		int b;
		int g;
		int r;
		int a;
	} channels;
	int components[4];
	int bits;
};

void memcpy(union Color *dest, __global uchar *src, size_t n, size_t start)
{
   
    int i;
   
   for (i = start; i < start + n; i++)
       dest[i] = (union Color)src[i];
}

uchar clp(uchar val, uchar min, uchar max) {
	return val  max ? max : val);
}

uchar round_to_5_bits(float val) {
	return clp(val * 31.0f / 255.0f + 0.5f, 0, 31);
}

uchar round_to_4_bits(float val) {
	return clp(val * 15.0f / 255.0f + 0.5f, 0, 15);
}





__constant static const int g_codeword_tables[8][4] = {
	{-8, -2, 2, 8},
	{-17, -5, 5, 17},
	{-29, -9, 9, 29},
	{-42, -13, 13, 42},
	{-60, -18, 18, 60},


	{-80, -24, 24, 80},
	{-106, -33, 33, 106},
	{-183, -47, 47, 183}};



__constant static const uchar g_mod_to_pix[4] = {3, 2, 0, 1};





















__constant static const uchar g_idx_to_num[4][8] = {
	{0, 4, 1, 5, 2, 6, 3, 7},        
	{8, 12, 9, 13, 10, 14, 11, 15},  
	{0, 4, 8, 12, 1, 5, 9, 13},      
	{2, 6, 10, 14, 3, 7, 11, 15}     
};




union Color makeColor(const union Color base, int lum) {
	int b = (int)(base.channels.b) + lum;
	int g = (int)(base.channels.g) + lum;
	int r = (int)(base.channels.r) + lum;
	union Color color;
	color.channels.b = (uchar)(clp(b, 0, 255));
	color.channels.g = (uchar)(clp(g, 0, 255));
	color.channels.r = (uchar)(clp(r, 0, 255));


	return color;
}



uint getColorError(const union Color u, const union Color v) {
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

void WriteColors444(uchar* block,
                    int new_dst,
					const union Color color0,
					const union Color color1) {
	
	block[new_dst] = (color0.channels.r & 0xf0) | (color1.channels.r >> 4);
	block[new_dst + 1] = (color0.channels.g & 0xf0) | (color1.channels.g >> 4);
	block[new_dst + 2] = (color0.channels.b & 0xf0) | (color1.channels.b >> 4);
}

void WriteColors555(__global uchar* block,


                    int new_dst,
				    const union Color color0,
					const union Color color1) {
	
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
	
	int delta_r =
    (int)(color1.channels.r >> 3) - (color0.channels.r >> 3);
	int delta_g =
	(int)(color1.channels.g >> 3) - (color0.channels.g >> 3);
	int delta_b =
	(int)(color1.channels.b >> 3) - (color0.channels.b >> 3);
	
	
	block[new_dst] = (color0.channels.r & 0xf8) | two_compl_trans_table[delta_r + 4];
	block[new_dst + 1] = (color0.channels.g & 0xf8) | two_compl_trans_table[delta_g + 4];
	block[new_dst + 2] = (color0.channels.b & 0xf8) | two_compl_trans_table[delta_b + 4];
}

void WriteCodewordTable(__global uchar* block,
                        int new_dst,
					    uchar sub_block_id,
						uchar table) {
	
	uchar shift = (2 + (3 - sub_block_id * 3));
	block[new_dst + 3] &= ~(0x07 << shift);
	block[new_dst + 3] |= table << shift;
}

void WritePixelData(__global uchar* block, int new_dst, uint pixel_data) {
	block[new_dst + 4] |= pixel_data >> 24;
	block[new_dst + 5] |= (pixel_data >> 16) & 0xff;
	block[new_dst + 6] |= (pixel_data >> 8) & 0xff;
	block[new_dst + 7] |= pixel_data & 0xff;
}

void WriteFlip(__global uchar* block, int new_dst, bool flip) {
	block[new_dst + 3] &= ~0x01;
	block[new_dst + 3] |= (uchar)(flip);
}

void WriteDiff(__global uchar* block, int new_dst, bool diff) {
	block[new_dst + 3] &= ~0x02;
	block[new_dst + 3] |= (uchar)((diff) << 1);
}

void ExtractBlock(uchar* dst, const uchar* src, int width) {
    int j;
	for (j = 0; j < 4; ++j) {
		memcpy(&dst[j * 4 * 4], src, 4 * 4, 0);
		src += width * 4;
	}
}





union Color makeColor444(const float* bgr) {
	uchar b4 = round_to_4_bits(bgr[0]);
	uchar g4 = round_to_4_bits(bgr[1]);
	uchar r4 = round_to_4_bits(bgr[2]);
	union Color bgr444;
	bgr444.channels.b = (b4 << 4) | b4;
	bgr444.channels.g = (g4 << 4) | g4;
	bgr444.channels.r = (r4 << 4) | r4;
	
	bgr444.channels.a = 0x44;
	return bgr444;
}





union Color makeColor555(const float* bgr) {
	uchar b5 = round_to_5_bits(bgr[0]);
	uchar g5 = round_to_5_bits(bgr[1]);
	uchar r5 = round_to_5_bits(bgr[2]);
	union Color bgr555;
	bgr555.channels.b = (b5 > 2);
	bgr555.channels.g = (g5 > 2);
	bgr555.channels.r = (r5 > 2);
	
	bgr555.channels.a = 0x55;
	return bgr555;
}
	
void getAverageColor(const union Color* src, float* avg_color)
{
	uchar sum_b = 0, sum_g = 0, sum_r = 0;
	uint i;
	for (i = 0; i < 8; ++i) {
		sum_b += src[i].channels.b;
		sum_g += src[i].channels.g;
		sum_r += src[i].channels.r;
	}
	
	const float kInv8 = 1.0f / 8.0f;
	avg_color[0] = (float)((sum_b) * kInv8);
	avg_color[1] = (float)((sum_g) * kInv8);
	avg_color[2] = (float)((sum_r) * kInv8);
}


bool tryCompressSolidBlock(__global uchar* dst,
                            int new_dst,
						   const union Color* src,
						   ulong* error)
{
    uint i, j, tbl_idx, mod_idx;
	for (uint i = 1; i < 16; ++i) {
		if (src[i].bits != src[0].bits)
			return false;
	}
	
	
	
    for (uint i = 0; i < 8; i ++) {
        dst[new_dst + i] = 0;
    }
	
	float src_color_float[3] = {(float)(src->channels.b),
		                        (float)(src->channels.g),
		                        (float)(src->channels.r)};
	union Color base = makeColor555(src_color_float);
	
	WriteDiff(dst, new_dst, true);
	WriteFlip(dst, new_dst, false);
	WriteColors555(dst, new_dst, base, base);
	
	uchar best_tbl_idx = 0;
	uchar best_mod_idx = 0;
	uchar best_mod_err = 2147483647; 
	
	
	
	for (tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		
		


		for (mod_idx = 0; mod_idx < 4; ++mod_idx) {
			int lum = g_codeword_tables[tbl_idx][mod_idx];
			const union Color color = makeColor(base, lum);
			
			int mod_err = getColorError(*src, color);
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
	
	WriteCodewordTable(dst, new_dst, 0, best_tbl_idx);
	WriteCodewordTable(dst, new_dst, 1, best_tbl_idx);
	
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
	
	WritePixelData(dst, new_dst, pix_data);
	*error = 16 * best_mod_err;
	return true;
}


ulong compressBlock(__global uchar* dst,
                    int new_dst,
	                const union Color* ver_src,
				    const union Color* hor_src,
					ulong threshold)
{
	ulong solid_error = 0;
	if (tryCompressSolidBlock(dst, new_dst, ver_src, &solid_error)) {
		return solid_error;
	}
	
	const union Color* sub_block_src[4] = {ver_src, ver_src + 8, hor_src, hor_src + 8};
	uint i, j, light_idx;
	union Color sub_block_avg[4];
	bool use_differential[2] = {true, true};
	
	
	for (i = 0, j = 1; i < 4; i += 2, j += 2) {
		float avg_color_0[3];
		getAverageColor(sub_block_src[i], avg_color_0);
		union Color avg_color_555_0 = makeColor555(avg_color_0);
		
		float avg_color_1[3];
		getAverageColor(sub_block_src[j], avg_color_1);
		union Color avg_color_555_1 = makeColor555(avg_color_1);
		
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
	
	
    return 0;
}

__kernel void kernel_id(__global float *output,
			__global int *width,
			__global int *height,
			__global uchar* src,
			__global uchar* dst)
{
	uint gid = get_global_id(0);
    int x = (gid / ((*width) / 4)) * 4;
    int y = (gid % ((*width) / 4)) * 4;
    size_t new_src = (*width) * 4 * 4 * (y / 4);
    
    int new_dst = 8 * (x / 4) * (y / 4);

	output[gid] = 0.0;

    union Color ver_blocks[16];
	union Color hor_blocks[16];
	
	unsigned long compressed_error = 0;
	
    
	
	
    
    memcpy(ver_blocks, src, 8, new_src + x * 4);
    memcpy(ver_blocks + 2, src, 8, new_src + x * 4 + (*width));
    memcpy(ver_blocks + 4, src, 8, new_src + x * 4 + (*width) * 2);
    memcpy(ver_blocks + 6, src, 8, new_src + x * 4 + (*width) * 3);
    memcpy(ver_blocks + 8, src, 8, new_src + x * 4 + 2);
    memcpy(ver_blocks + 2, src, 8, new_src + x * 4 + (*width) + 2);
    memcpy(ver_blocks + 2, src, 8, new_src + x * 4 + (*width) * 2 + 2);
    memcpy(ver_blocks + 2, src, 8, new_src + x * 4 + (*width) * 3 + 2);
    
    memcpy(hor_blocks, src, 16, new_src + x * 4);
    memcpy(hor_blocks, src, 16, new_src + x * 4 + (*width));
    memcpy(hor_blocks, src, 16, new_src + x * 4 + (*width) * 2);
    memcpy(hor_blocks, src, 16, new_src + x * 4 + (*width) * 3);
	
    output[gid] = compressBlock(dst, new_dst, ver_blocks, hor_blocks, 2147483647);
    
    

}

#include 
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


int CL_COMPILE_ERR(int cl_ret,
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


void read_kernel(string file_name, string &str_kernel)
{
	ifstream in_file(file_name.c_str());
	in_file.open(file_name.c_str());
	DIE( !in_file.is_open(), "open file" );

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


void cl_get_compiler_err_log(cl_program program,
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
#ifndef CL_HELPER_H
#define CL_HELPER_H

#include 

using namespace std;

int CL_ERR(int cl_ret);
int CL_COMPILE_ERR(int cl_ret,
                  cl_program program,
                  cl_device_id device);

const char* cl_get_string_err(cl_int err);
void cl_get_compiler_err_log(cl_program program,
                             cl_device_id device);

void read_kernel(string file_name, string &str_kernel);

#define DIE(assertion, call_description)                    \
do {                                                        \
    if (assertion) {                                        \
            fprintf(stderr, "(%d): ",                       \
                            __LINE__);                      \
            perror(call_description);                       \
            exit(EXIT_FAILURE);                             \
    }                                                       \
} while(0);

#endif

#include 
#include 
#include 
#include 
#include 
#include 
#include "helper.hpp"

#include "compress.hpp"

using namespace std;


void gpu_find(cl_device_id &device)
{
	cl_platform_id platform;
	cl_uint platform_num = 0;
	cl_platform_id* platform_list = NULL;

	cl_uint device_num = 0;
	cl_device_id* device_list = NULL;

	size_t attr_size = 0;
	cl_char* attr_data = NULL;

	
	CL_ERR( clGetPlatformIDs(0, NULL, &platform_num));
	platform_list = new cl_platform_id[platform_num];
	DIE(platform_list == NULL, "alloc platform_list");

	
	CL_ERR( clGetPlatformIDs(platform_num, platform_list, NULL));
	

	
	for(uint i=0; i<platform_num; i++)
	{
		
		CL_ERR( clGetPlatformInfo(platform_list[i],
				CL_PLATFORM_VENDOR, 0, NULL, &attr_size));
		attr_data = new cl_char[attr_size];
		DIE(attr_data == NULL, "alloc attr_data");

		
		CL_ERR( clGetPlatformInfo(platform_list[i],
				CL_PLATFORM_VENDOR, attr_size, attr_data, NULL));
	

		
		if(string((const char*)attr_data).find("NVIDIA", 0) != string::npos)
			platform = platform_list[i]; 
		delete[] attr_data;

		
		CL_ERR( clGetPlatformInfo(platform_list[i],
				CL_PLATFORM_VERSION, 0, NULL, &attr_size));
		attr_data = new cl_char[attr_size];
		DIE(attr_data == NULL, "alloc attr_data");

		
		CL_ERR( clGetPlatformInfo(platform_list[i],
				CL_PLATFORM_VERSION, attr_size, attr_data, NULL));
		
		delete[] attr_data;
	}

	
	DIE(platform == 0, "platform selection");

	
	CL_ERR( clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &device_num));
	device_list = new cl_device_id[device_num];
	DIE(device_list == NULL, "alloc devices");

	
	CL_ERR( clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU,
		  device_num, device_list, NULL));
	

	
	for(uint i=0; i<device_num; i++)
	{
		
		CL_ERR( clGetDeviceInfo(device_list[i], CL_DEVICE_NAME,
				0, NULL, &attr_size));
		attr_data = new cl_char[attr_size];
		DIE(attr_data == NULL, "alloc attr_data");

		
		CL_ERR( clGetDeviceInfo(device_list[i], CL_DEVICE_NAME,
				attr_size, attr_data, NULL));
		
		delete[] attr_data;

		
		CL_ERR( clGetDeviceInfo(device_list[i], CL_DEVICE_VERSION,
				0, NULL, &attr_size));
		attr_data = new cl_char[attr_size];
		DIE(attr_data == NULL, "alloc attr_data");

		
		CL_ERR( clGetDeviceInfo(device_list[i], CL_DEVICE_VERSION,
				attr_size, attr_data, NULL));
		
		delete[] attr_data;
	}

	
	if(device_num > 0)
		device = device_list[0];
	else
		device = 0;

	delete[] platform_list;
	delete[] device_list;
}

TextureCompressor::TextureCompressor() { 
    cl_device_id device;
    gpu_find(device);
    DIE(device == 0, "check valid device");
    this->device = device;

} 	
TextureCompressor::~TextureCompressor() { }	





void gpu_execute_kernel(cl_device_id device, int width, int height,
                        const uint8_t* src, uint8_t* dst)
{
    


	cl_int ret;
	cl_context context;
	cl_command_queue cmd_queue;
	cl_program program;
	cl_kernel kernel;

	string kernel_src;

	
	context = clCreateContext(0, 1, &device, NULL, NULL, &ret);
	CL_ERR( ret );

	
	cmd_queue = clCreateCommandQueue(context, device, 0, &ret);
	CL_ERR( ret );

	
	size_t size = (width * height) / 16;
	cl_long *buf_host = new cl_long[size];
	DIE ( buf_host == NULL, "alloc buf_host1" );

	
    
	
	read_kernel("device.cl", kernel_src);
	const char* kernel_c_str = kernel_src.c_str();

	
	program = clCreateProgramWithSource(context, 1,
		  &kernel_c_str, NULL, &ret);
	CL_ERR( ret );

	
	ret = clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", NULL, NULL);
	CL_COMPILE_ERR( ret, program, device );

	
	kernel = clCreateKernel(program, "kernel_id", &ret);
	CL_ERR( ret );

	
	
	
	
	
	cl_mem buf_dev = clCreateBuffer(context, CL_MEM_READ_WRITE,
				  sizeof(cl_long) * size, NULL, &ret);
	CL_ERR( ret );
	cl_mem width_dev = clCreateBuffer(context, CL_MEM_READ_WRITE,
				  sizeof(cl_int), NULL, &ret);
	CL_ERR( ret );
	cl_mem height_dev = clCreateBuffer(context, CL_MEM_READ_WRITE,
				  sizeof(cl_int), NULL, &ret);
	CL_ERR( ret );
    cl_mem src_dev = clCreateBuffer(context, CL_MEM_READ_WRITE,
				  sizeof(cl_uchar), NULL, &ret);
	CL_ERR( ret );
	cl_mem dst_dev = clCreateBuffer(context, CL_MEM_READ_WRITE,
				  sizeof(cl_uchar), NULL, &ret);
	CL_ERR( ret );

	
	
	CL_ERR( clEnqueueWriteBuffer(cmd_queue, width_dev, CL_TRUE, 0,
		  sizeof(int), &width, 0, NULL, NULL));
	CL_ERR( clEnqueueWriteBuffer(cmd_queue, height_dev, CL_TRUE, 0,
		  sizeof(int), &height, 0, NULL, NULL));
	CL_ERR( clEnqueueWriteBuffer(cmd_queue, src_dev, CL_TRUE, 0,
		  sizeof(uint8_t), src , 0, NULL, NULL));
	CL_ERR( clEnqueueWriteBuffer(cmd_queue, dst_dev, CL_TRUE, 0,
		  sizeof(uint8_t), dst , 0, NULL, NULL));

	
	
	CL_ERR( clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &buf_dev) );
	CL_ERR( clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &width_dev) );
	CL_ERR( clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &height_dev) );
	CL_ERR( clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &src_dev) );
	CL_ERR( clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *) &dst_dev) );

	
	
	
	size_t globalSize[2] = {size, 0};
	size_t localSize[2] = {1, 0};
    
	ret = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL,
		  globalSize, localSize, 0, NULL, NULL);
	CL_ERR( ret );

	
	CL_ERR( clEnqueueReadBuffer(cmd_queue, buf_dev, CL_TRUE, 0,
		  sizeof(float) * size, buf_host, 0, NULL, NULL));

	
	
	

	
	CL_ERR( clFinish(cmd_queue) );

	
	CL_ERR( clReleaseMemObject(buf_dev) );
	CL_ERR( clReleaseMemObject(width_dev) );
	CL_ERR( clReleaseMemObject(height_dev) );
	CL_ERR( clReleaseMemObject(src_dev) );
	CL_ERR( clReleaseMemObject(dst_dev) );
	CL_ERR( clReleaseCommandQueue(cmd_queue) );
	CL_ERR( clReleaseContext(context) );

	
	delete[] buf_host;
}

unsigned long TextureCompressor::compress(const uint8_t* src,
									  uint8_t* dst,
									  int width,
									  int height)
{
    gpu_execute_kernel(device, width, height,
                        src, dst);

    return 0;
}
