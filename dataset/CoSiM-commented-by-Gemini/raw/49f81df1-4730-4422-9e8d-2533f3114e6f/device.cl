
union Color {
	struct BgraColorType {
		uchar b;
		uchar g;
		uchar r;
		uchar a;
	} channels;
	uchar components[4];
	int bits;
};

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

uchar clamp_m(int val, int min, int max) {
	uchar return_value;
	int result = val  max ? max : val);
	if (result > 255) {
		return_value = 255;
	} else {
		return_value = result;
	}

	return return_value;
}

void my_memset(__global uchar *buf, uchar value, int count)
{
	int i;
	for (i = 0; i < count; i++) {
		buf[i] = value;
	}
}

void my_memcpy(void *dest, __global void *src, int count)
{
	int i;
	char *aux_dest = dest;
	__global char *aux_src = src;
	for (i = 0; i < count; i++) {
		*aux_dest = *aux_src;
		aux_dest++;
		aux_src++;
	}
}

uchar round_to_5_bits(float val) {
	return clamp_m((int)(val * 31.0f / 255.0f + 0.5f), 0, 31);
}

uchar round_to_4_bits(float val) {


	return clamp_m((int)(val * 15.0f / 255.0f + 0.5f), 0, 15);
}

union Color makeColor(union Color base, short lum) {
	int b = (int)(base.channels.b) + lum;
	int g = (int)(base.channels.g) + lum;
	int r = (int)(base.channels.r) + lum;
	union Color color;
	color.channels.b = (uchar)(clamp_m(b, 0, 255));
	color.channels.g = (uchar)(clamp_m(g, 0, 255));
	color.channels.r = (uchar)(clamp_m(r, 0, 255));
	return color;
}



uint getColorError(union Color u, union Color v) {
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
	
	ushort delta_r =
	(ushort)(color1.channels.r >> 3) - (color0.channels.r >> 3);
	ushort delta_g =
	(ushort)(color1.channels.g >> 3) - (color0.channels.g >> 3);
	ushort delta_b =
	(ushort)(color1.channels.b >> 3) - (color0.channels.b >> 3);
	
	
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

void WritePixelData(__global uchar *block, uint pixel_data) {
	block[4] |= pixel_data >> 24;
	block[5] |= (pixel_data >> 16) & 0xff;
	block[6] |= (pixel_data >> 8) & 0xff;
	block[7] |= pixel_data & 0xff;
}

void WriteFlip(__global uchar *block, bool flip) {
	block[3] &= ~0x01;
	block[3] |= (uchar)(flip);
}

void WriteDiff(__global uchar *block, bool diff) {
	block[3] &= ~0x02;
	block[3] |= (uchar)(diff) << 1;
}

void ExtractBlock(__global uchar *dst, __global uchar *src, int width) {
	int i;
	for (int j = 0; j < 4; ++j) {
		for (i=0; i < 16; i++) {


			(&dst[j*16])[i] = src[i];
		}
		src += width * 4;
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
	uint sum_b = 0, sum_g = 0, sum_r = 0;
	
	for (unsigned int i = 0; i < 8; ++i) {
		sum_b += src[i].channels.b;


		sum_g += src[i].channels.g;
		sum_r += src[i].channels.r;
	}
	
	float kInv8 = 1.0f / 8.0f;
	avg_color[0] = (float)(sum_b) * kInv8;
	avg_color[1] = (float)(sum_g) * kInv8;
	avg_color[2] = (float)(sum_r) * kInv8;
}

unsigned long computeLuminance(__global uchar *block,
						   union Color *src,


						   union Color base,
						   int sub_block_id,
						   __constant uchar *idx_to_num_tab,
						   unsigned long threshold)
{
	uint best_tbl_err = threshold;
	uchar best_tbl_idx = 0;
	uchar best_mod_idx[8][8];  

	
	
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		
		
		union Color candidate_color[4];  
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			ushort lum = g_codeword_tables[tbl_idx][mod_idx];
			candidate_color[mod_idx] = makeColor(base, lum);
		}
		
		uint tbl_err = 0;
		
		for (unsigned int i = 0; i < 8; ++i) {
			
			
			uint best_mod_err = threshold;
			for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
				union Color color = candidate_color[mod_idx];
				
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


bool tryCompressSolidBlock(__global uchar *dst,
						   union Color *src,
						   unsigned long *error)
{
	for (unsigned int i = 1; i < 16; ++i) {
		if (src[i].bits != src[0].bits)
			return false;
	}
	
	
	my_memset(dst, 0, 8);
	
	float src_color_float[3] = {(float)(src->channels.b),
		(float)(src->channels.g),
		(float)(src->channels.r)};
	union Color base = makeColor555(src_color_float);
	
	WriteDiff(dst, true);
	WriteFlip(dst, false);
	WriteColors555(dst, base, base);
	
	uchar best_tbl_idx = 0;
	uchar best_mod_idx = 0;
	uint best_mod_err = 2147483647 * 2 + 1; 
	
	
	
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		
		
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			ushort lum = g_codeword_tables[tbl_idx][mod_idx];
			union Color color = makeColor(base, lum);
			
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

	return true;
}



unsigned long compressBlock(__global uchar *dst,
							union Color *ver_src,
							union Color *hor_src,
							unsigned long threshold)
{
	unsigned long solid_error = 0;
	if (tryCompressSolidBlock(dst, ver_src, &solid_error)) {
		return solid_error;
	}
	
	union Color *sub_block_src[4] = {ver_src, ver_src + 8, hor_src, hor_src + 8};
	
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
	
	
	my_memset(dst, 0, 8);
	
	WriteDiff(dst, use_differential[!!flip]);
	WriteFlip(dst, flip);
	
	uchar sub_block_off_0 = flip ? 2 : 0;
	uchar sub_block_off_1 = sub_block_off_0 + 1;
	
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

__kernel void compress(int NWidth,
				int NHeight,
				__global uchar* S,
				__global uchar* D)
{
	int i = get_global_id(0);
	int j = get_global_id(1);
    
    union Color ver_blocks[16];
	 union Color hor_blocks[16];
	
	int y = i * 4;
	int x = j * 4;
	int dest_offset = j * 8;
	int src_offset = i * NWidth * 4 * 4;



	__global union Color* row0 = (__global union Color*)(((S + src_offset)) + x * 4);
	__global union Color* row1 = row0 + NWidth;
	__global union Color* row2 = row1 + NWidth;
	__global union Color* row3 = row2 + NWidth;
	
	my_memcpy(ver_blocks, row0, 8);
	my_memcpy(ver_blocks + 2, row1, 8);
	my_memcpy(ver_blocks + 4, row2, 8);
	my_memcpy(ver_blocks + 6, row3, 8);
	my_memcpy(ver_blocks + 8, row0 + 2, 8);
	my_memcpy(ver_blocks + 10, row1 + 2, 8);
	my_memcpy(ver_blocks + 12, row2 + 2, 8);
	my_memcpy(ver_blocks + 14, row3 + 2, 8);
	
	my_memcpy(hor_blocks, row0, 16);
	my_memcpy(hor_blocks + 4, row1, 16);
	my_memcpy(hor_blocks + 8, row2, 16);
	my_memcpy(hor_blocks + 12, row3, 16);
	
	compressBlock(D + dest_offset, ver_blocks, hor_blocks, 2147483647);
}

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

#if __APPLE__
#else
#endif

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

#include "compress.hpp"
#include "helper.hpp"

using namespace std;

TextureCompressor::TextureCompressor() { } 	
TextureCompressor::~TextureCompressor() { }	


void gpu_find(cl_device_id &device,
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
		
		
		CL_ERR( clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &device_num));
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
			
			
			
			if((device_num > 1) && (dev == device_select)){
				device = device_list[dev];
				cout << " <--- SELECTED ";
			}
			
			cout << endl;
		}
	}
	
	delete[] platform_list;
	delete[] device_list;
}

void gpu_profile_kernel(const uint8_t* src, uint8_t* dst, int width, int height, cl_device_id device_id)
{
	string			kernel_src;
	cl_context		context;
	cl_command_queue commands;
	cl_program		program;
	cl_kernel		kernel;
	cl_mem			s_in;
	cl_mem			d_out;
	int				i, ret;
	int 			NWidth = width;
	int 			NHeight = height;

	
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &ret);
	CL_ERR( ret );
	
	commands = clCreateCommandQueue(context, device_id,
									0, &ret);
	CL_ERR( ret );
	
	
	
	
	
	s_in   = clCreateBuffer(context,  CL_MEM_READ_ONLY,
							sizeof(unsigned char) * 4 * width * height, NULL, NULL);
	d_out  = clCreateBuffer(context,  CL_MEM_WRITE_ONLY,
							sizeof(unsigned char) * width * height / 2, NULL, NULL);
	
	
	read_kernel("device.cl", kernel_src);
	const char* kernel_c_str = kernel_src.c_str();
	
	
	program = clCreateProgramWithSource(context, 1,
				(const char **) &kernel_c_str, NULL, &ret);
	CL_ERR( ret );
	
	
	ret = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	CL_COMPILE_ERR( ret, program, device_id );
	
	
	kernel = clCreateKernel(program, "compress", &ret);
	CL_ERR( ret );
	
	
	ret  = 0;
	ret  = clSetKernelArg(kernel, 0, sizeof(int), &NWidth);
	ret |= clSetKernelArg(kernel, 1, sizeof(int), &NHeight);
	ret |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &s_in);
	ret |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_out);

	
	ret = clEnqueueWriteBuffer(commands, s_in, CL_TRUE, 0,
							   sizeof(unsigned char) * 4 * NHeight * NWidth, src, 0, NULL, NULL);
	cl_event prof_event;

	
	size_t global_size[2] = {(size_t) (height/4), (size_t) (width/4)};
	
	ret = clEnqueueNDRangeKernel(commands, kernel, 2, NULL,
								 global_size, 0, 0, NULL, &prof_event);

	
	clFinish(commands);

	
	ret = clEnqueueReadBuffer( commands, d_out, CL_TRUE, 0,
							  sizeof(float) * height * width / 2, dst, 0, NULL, NULL );
	
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseMemObject(s_in);
	clReleaseMemObject(d_out);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);
}

unsigned long TextureCompressor::compress(const uint8_t* src,
									  uint8_t* dst,
									  int width,
									  int height)
{
	cl_device_id device;
	int device_select = 1;

	gpu_find(device, device_select);
	gpu_profile_kernel(src, dst, width, height, device);

	return 0;
}
