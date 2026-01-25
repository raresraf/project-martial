
#ifndef CL_HELPER_H
#define CL_HELPER_H

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
#include 
#include 
#include 

using namespace std;


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

#define DIE(assertion, call_description)  \
do { \
	if (assertion) { \
		fprintf(stderr, "(%d): ", __LINE__); \
		perror(call_description); \
		exit(EXIT_FAILURE); \
	} \
} while(0);


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


#endif
union Color {
	struct BgraColorType {
		uchar b;
		uchar g;
		uchar r;
		uchar a;
	} channels;
	uchar components[4];
	uint bits;
};

#define ALIGNAS(X)	__attribute__((aligned(X)))

uchar my_clamp(int val, int mini, int maxi)
{
	return (uchar)(val  maxi ? maxi : val));
}

uchar round_to_5_bits(float val) {
	return my_clamp(val * 31.0f / 255.0f + 0.5f, 0, 31);
}

uchar round_to_4_bits(float val) {
	return my_clamp(val * 15.0f / 255.0f + 0.5f, 0, 15);
}

__constant ALIGNAS(16) unsigned long g_codeword_tables[8][4] = {
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

union Color makeColor(union Color base, short lum) {
	union Color color;
	int b = convert_int(base.channels.b) + lum,
		g = convert_int(base.channels.g) + lum,
		r = convert_int(base.channels.r) + lum;

	color.channels.b = convert_int(my_clamp(b, 0, 255));
	color.channels.g = convert_int(my_clamp(g, 0, 255));
	color.channels.r = convert_int(my_clamp(r, 0, 255));
	
	return color;
}

int getColorErrorMetric(union Color u, union Color v) {
	float delta_b = (float)(u.channels.b) - v.channels.b;
	float delta_g =(float)(u.channels.g) - v.channels.g;
	float delta_r = (float)(u.channels.r) - v.channels.r;
	return (unsigned long)(0.299f * delta_b * delta_b +
								 0.587f * delta_g * delta_g +
								 0.114f * delta_r * delta_r);
}

unsigned long getColorError(union Color u, union Color v) {
	int delta_b = (int)(u.channels.b) - v.channels.b;
	int delta_g = (int)(u.channels.g) - v.channels.g;
	int delta_r = (int)(u.channels.r) - v.channels.r;

	return delta_b * delta_b + delta_g * delta_g + delta_r * delta_r;
}

void WriteColors444(__global uchar *block,
						   union Color color0,
						   union Color color1) {
	
	block[0] = (color0.channels.r & 0xf0) | (color1.channels.r >> 4);
	block[1] = (color0.channels.g & 0xf0) | (color1.channels.g >> 4);
	block[2] = (color0.channels.b & 0xf0) | (color1.channels.b >> 4);
}

void WriteColors555(__global uchar *block,
						   union Color color0,
						   union Color color1) {
	
	uchar two_compl_trans_table[8] = {
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

void WriteCodewordTable(__global uchar *block,
							   uchar sub_block_id,
							   uchar table) {
	
	uchar shift = (2 + (3 - sub_block_id * 3));
	block[3] &= ~(0x07 << shift);
	block[3] |= table << shift;
}



void WritePixelData(__global uchar *block, unsigned int pixel_data) {
	block[4] |= pixel_data >> 24;
	block[5] |= (pixel_data >> 16) & 0xff;
	block[6] |= (pixel_data >> 8) & 0xff;
	block[7] |= pixel_data & 0xff;
}

void WriteFlip(__global uchar *block, bool flip) {
	block[3] &= ~0x01;
	block[3] |= convert_uchar(flip);
}

void WriteDiff(__global uchar *block, bool diff) {
	block[3] &= ~0x02;
	block[3] |= convert_uchar(diff) << 1;
}

void ExtractBlock(__global uchar *dst, uchar *src, int width) {
	for (int j = 0; j < 4; ++j) {
		int offset = width *4;
		for (int k = 0; k < 4 * 4; k++) {
			dst[j * 4 * 4 + k] = *(src + k);
		}
		src += offset;
	}
}

union Color makeColor444(float *bgr) {
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

union Color makeColor555(float *bgr) {
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


void getAverageColor(union Color *src, float *avg_color)
{
	unsigned int sum_b = 0, sum_g = 0, sum_r = 0;
	
	for (unsigned int i = 0; i < 8; ++i) {
		sum_b += src[i].channels.b;
		sum_g += src[i].channels.g;
		sum_r += src[i].channels.r;
	}
	
	float kInv8 = 1.0f / 8.0f;
	avg_color[0] = convert_float(sum_b) * kInv8;
	avg_color[1] = convert_float(sum_g) * kInv8;
	avg_color[2] = convert_float(sum_r) * kInv8;
}


unsigned long computeLuminance(__global uchar *block,
						   union Color *src,
						   union Color base,
						   int sub_block_id,
						   uchar *idx_to_num_tab,
						   unsigned long threshold, int param)
{
	int best_tbl_err = threshold;
	uchar best_tbl_idx = 0;
	uchar best_mod_idx[8][8];  

	
	
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		
		
		union Color candidate_color[4];  


		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			candidate_color[mod_idx] = makeColor(base, lum);
		}
		
		unsigned int tbl_err = 0;
		
		for (unsigned int i = 0; i < 8; ++i) {
			
			
			unsigned int best_mod_err = threshold;
			for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
				union Color color = candidate_color[mod_idx];
				
				unsigned int mod_err = (param == 1) ? 
					getColorErrorMetric(src[i], color) :
					getColorError(src[i], color);
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

	for (unsigned int i = 0; i < 8; ++i) {
		uchar mod_idx = best_mod_idx[best_tbl_idx][i];


		uchar pix_idx = g_mod_to_pix[mod_idx];
		
		unsigned int lsb = pix_idx & 0x1;
		unsigned int msb = pix_idx >> 1;
		
		
		int texel_num = idx_to_num_tab[i];
		pix_data |= msb << (texel_num + 16);
		pix_data |= lsb << (texel_num);
	}

	WritePixelData(block, pix_data);

	return best_tbl_err;
}

bool tryCompressSolidBlock(__global uchar *dst,
						   union Color *src,
						   unsigned long *error, int param)
{
	for (unsigned int i = 1; i < 16; ++i) {
		if (src[i].bits != src[0].bits)
			return false;
	}
	
	
	for (unsigned int i = 0; i < 8; i++) {
		dst[i] = 0;
	}
	
	float src_color_float[3] = {convert_float(src->channels.b),
		convert_float(src->channels.g),
		convert_float(src->channels.r)};
	union Color base = makeColor555(src_color_float);
	
	WriteDiff(dst, true);
	WriteFlip(dst, false);
	WriteColors555(dst, base, base);
	
	uchar best_tbl_idx = 0;
	uchar best_mod_idx = 0;


	unsigned int best_mod_err = UINT_MAX; 
	
	
	
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		
		
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			union Color color = makeColor(base, lum);
			
			unsigned int mod_err = (param == 1) ? 
				getColorErrorMetric(*src, color) : getColorError(*src, color);
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
	unsigned int lsb = pix_idx & 0x1;
	unsigned int msb = pix_idx >> 1;
	
	unsigned int pix_data = 0;
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


unsigned long compressBlock(__global uchar *dst,
						   union Color *ver_src,
						   union Color *hor_src,
						   unsigned long threshold, int param)
{
	unsigned long solid_error = 0;


	if (tryCompressSolidBlock(dst, ver_src, &solid_error, param)) {
		return solid_error;
	}
	
	union Color* sub_block_src[4] = {ver_src, ver_src + 8, hor_src, hor_src + 8};
	
	union Color sub_block_avg[4];
	bool use_differential[2] = {true, true};
	
	uchar sub_block_param[4];

	
	
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
	
	
	
	
	unsigned int sub_block_err[4] = {0};
	for (unsigned int i = 0; i < 4; ++i) {
		for (unsigned int j = 0; j < 8; ++j) {
			sub_block_err[i] += 
				(param == 1) ? 
				getColorErrorMetric(sub_block_avg[i], sub_block_src[i][j])
				: getColorError(sub_block_avg[i], sub_block_src[i][j]);
		}
	}
	
	bool flip =
	sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1];
	
	
	
	for (unsigned int i = 0; i < 8; i++) {
		dst[i] = '\0';
	}
	
	WriteDiff(dst, use_differential[!!flip]);
	WriteFlip(dst, flip);
	
	uchar sub_block_off_0 = flip ? 2 : 0;
	uchar sub_block_off_1 = sub_block_off_0 + 1;
	
	for (int k = 0; k < 4; k++) {
		sub_block_param[k] = g_idx_to_num[sub_block_off_0][k];
	}

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
								   sub_block_param,
								   threshold, param);

	for (int k = 0; k < 4; k++)
	{
		sub_block_param[k] = g_idx_to_num[sub_block_off_1][k];
	}

	
	lumi_error2 = computeLuminance(dst, sub_block_src[sub_block_off_1],
								   sub_block_avg[sub_block_off_1], 1,
								   sub_block_param,
								   threshold, param);
	
	return lumi_error1 + lumi_error2;
}

void multiple_memcpy(__private union Color dest[],__global union Color *src, int nr_colors) {
	for (int i = 0; i < nr_colors; i++) {
		union Color curr_src = *(src + i);
		dest[i].channels.b = curr_src.channels.b;
		dest[i].channels.g = curr_src.channels.g;
		dest[i].channels.r = curr_src.channels.r;
		dest[i].channels.a = curr_src.channels.a;

		for (int j = 0; j < 4; j++) {
			dest[i].components[j] = curr_src.components[j];
		}

		dest[i].bits = curr_src.bits;
	}
}

__kernel void kernel_device(__global int *src,
	__global int *dst, int width, int height, int param,
	__global unsigned long *compressed_error)
{
	union Color ver_blocks[16];
	union Color hor_blocks[16];

	int row = get_global_id(1);
	int col = get_global_id(0);

	__global union Color *row0 =  (__global union Color*) (src + row * width / 4 * 64 + col * 16), 
		*row1 = row0 + width * 4,
		*row2 = row1 + width * 4,
		*row3 = row2 + width * 4;

	multiple_memcpy(ver_blocks, row0, 2);
	multiple_memcpy((ver_blocks + 2), row1, 2);
	multiple_memcpy((ver_blocks + 4), row2, 2);
	multiple_memcpy((ver_blocks + 6), row3, 2);
	multiple_memcpy((ver_blocks + 8), row0 + 2, 2);
	multiple_memcpy((ver_blocks + 10), row1, 2);
	multiple_memcpy((ver_blocks) + 12, row2 + 2, 2);
	multiple_memcpy((ver_blocks + 14), row3 + 2, 2);
	multiple_memcpy((hor_blocks), row0, 2);
	multiple_memcpy((hor_blocks + 4), row1, 4);
	multiple_memcpy((hor_blocks + 8), row2, 4);
	multiple_memcpy((hor_blocks + 12), row3, 4);

	*compressed_error = *compressed_error + compressBlock((__global uchar*)(dst + 
		(row * width / 4 * 8 + col * 8)), 
		ver_blocks, hor_blocks, INT_MAX, param);
} >>>> file: texture_compress_skl.cpp
#include "compress.hpp"
#include "helper.hpp"


void gpu_find(cl_device_id &device, 
		uint platform_select, 
		uint device_select)
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
			CL_DEVICE_TYPE_GPU, 0, NULL, &device_num) == CL_DEVICE_NOT_FOUND) {
			device_num = 0;
			continue;
		}

		device_list = new cl_device_id[device_num];
		DIE(device_list == NULL, "alloc devices");

		
		CL_ERR( clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU,
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

			device = device_list[dev];

			cout << endl;
		}
	}

	delete[] platform_list;
	delete[] device_list;
}


unsigned long gpu_compress_block_kernel(cl_device_id device, 
	const char* kernel_name, int width, int height, const uint8_t *src, 
	uint8_t *dst)
{
	cl_int ret;
	unsigned long compressed_error = 0;

	cl_context context;
	cl_command_queue cmdQueue;
	cl_program program;
	cl_kernel kernel;

	string kernel_src;

	
	context = clCreateContext(0, 1, &device, NULL, NULL, &ret);
	CL_ERR( ret );

	
	cmdQueue = clCreateCommandQueue(context, device, 0, &ret);
	CL_ERR( ret );

	int buf_size = width * height * 4;

	cl_mem bufSrc = clCreateBuffer(context, CL_MEM_READ_ONLY,
		buf_size, NULL, &ret);
	CL_ERR( ret );
	cl_mem bufDst = clCreateBuffer(context, CL_MEM_READ_WRITE,
		buf_size / 8, NULL, &ret);
	CL_ERR( ret );
	cl_mem bufError = clCreateBuffer(context, CL_MEM_READ_WRITE,
		sizeof(unsigned long), NULL, &ret);
	CL_ERR(ret);

	
	CL_ERR( clEnqueueWriteBuffer(cmdQueue, bufSrc, CL_TRUE, 0,
		buf_size, src, 0, NULL, NULL));

	
	read_kernel("kernel.cl", kernel_src);
	const char* kernel_c_str = kernel_src.c_str();
	
	
	program = clCreateProgramWithSource(context, 1,
		  &kernel_c_str, NULL, &ret);
	CL_ERR( ret );

	
	ret = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	CL_COMPILE_ERR( ret, program, device );

	
	kernel = clCreateKernel(program, kernel_name, &ret);
	CL_ERR( ret );

	int param;
#ifdef USE_PERCEIVED_ERROR_METRIC
	param = 1;
#else
	param = 0;
#endif

	
	CL_ERR( clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&bufSrc) );
	CL_ERR( clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&bufDst) );
	CL_ERR( clSetKernelArg(kernel, 2, sizeof(cl_uint), (void *)&width) );
	CL_ERR( clSetKernelArg(kernel, 3, sizeof(cl_uint), (void *)&height) );
	CL_ERR( clSetKernelArg(kernel, 4, sizeof(cl_int), (void *)¶m) );
	CL_ERR( clSetKernelArg(kernel, 5, sizeof(cl_ulong), 
		(void *)&bufError) );

	
	size_t globalSize[2] = {(size_t)(width / 4), (size_t)(height / 4)};

	ret = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL,
		globalSize, 0, 0, NULL, NULL);
	CL_ERR( ret );

	
	CL_ERR( clEnqueueReadBuffer(cmdQueue, bufDst, CL_TRUE, 0,
		buf_size / 8, dst, 0, NULL, NULL));


	CL_ERR( clEnqueueReadBuffer(cmdQueue, bufError, CL_TRUE, 0,
		sizeof(unsigned long), &compressed_error, 0, NULL, NULL));

	
	CL_ERR( clFinish(cmdQueue) );

	
	CL_ERR( clReleaseMemObject(bufSrc) );
	CL_ERR( clReleaseMemObject(bufDst) );
	CL_ERR( clReleaseMemObject(bufError) );
	CL_ERR( clReleaseCommandQueue(cmdQueue) );
	CL_ERR( clReleaseContext(context) );

	return compressed_error;
}

unsigned long TextureCompressor::compress(const uint8_t* src,
									  uint8_t* dst,
									  int width,
									  int height)
{	
	unsigned long compressed_error = 0;

	compressed_error = gpu_compress_block_kernel(this->device, "kernel_device", 
		width, height, src, dst);
				
	return compressed_error;
}

TextureCompressor::TextureCompressor() {
	int platform_select = 0;
	int device_select = 0;

	gpu_find(this->device, platform_select, device_select);
 }	

TextureCompressor::~TextureCompressor() { }
