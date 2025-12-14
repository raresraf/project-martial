
#define ALIGNAS(X)	__attribute__((aligned(X)))
#define UINT32_MAX  (0xffffffff)
#define INT32_MAX (0x7fffffff)

union Color{
	struct BgraColorType {
		unsigned char b;
		unsigned char g;
		unsigned char r;
		unsigned char a;
	} channels;
	unsigned char components[4];
	unsigned int bits;
};

void my_memcpy (__global unsigned char *dest, __global unsigned char *src, size_t n) {
	while(n--) {
		*dest++ = *src++;
	}
}
void my_memset(__global unsigned char *s, int c, size_t n)
{
    while(n--) {
        *s++ = (unsigned char)c;
	}
}

 unsigned char my_clamp(unsigned char val, unsigned char min, unsigned char max) {
	return val  max ? max : val);
}

 unsigned char round_to_5_bits(float val) {
	return my_clamp(val * 31.0f / 255.0f + 0.5f, 0, 31);
}

 unsigned char round_to_4_bits(float val) {
	return my_clamp(val * 15.0f / 255.0f + 0.5f, 0, 15);
}


ALIGNAS(16) __constant short g_codeword_tables[8][4] = {
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


 union Color makeColor(const union Color base, short lum) {
	int b = (int)(base.channels.b) + lum;
	int g = (int)(base.channels.g) + lum;
	int r = (int)(base.channels.r) + lum;
	union Color color;
	color.channels.b = (unsigned char)(my_clamp(b, 0, 255));
	color.channels.g = (unsigned char)(my_clamp(g, 0, 255));
	color.channels.r = (unsigned char)(my_clamp(r, 0, 255));
	return color;
}



 unsigned int getColorError(const union Color u, const union Color v) {
#ifdef USE_PERCEIVED_ERROR_METRIC
	float delta_b = (float)(u.channels.b - v.channels.b);
	float delta_g = (float)(u.channels.g - v.channels.g);
	float delta_r = (float)(u.channels.r - v.channels.r);
	return (unsigned int)(0.299f * delta_b * delta_b +
								 0.587f * delta_g * delta_g +
								 0.114f * delta_r * delta_r);
#else
	int delta_b = (int)(u.channels.b - v.channels.b);
	int delta_g = (int)(u.channels.g - v.channels.g);
	int delta_r = (int)(u.channels.r - v.channels.r);
	return delta_b * delta_b + delta_g * delta_g + delta_r * delta_r;
#endif
}

 void WriteColors444(__global unsigned char* block,
						   const union Color color0,
						   const union Color color1) {
	
	block[0] = (color0.channels.r & 0xf0) | (color1.channels.r >> 4);
	block[1] = (color0.channels.g & 0xf0) | (color1.channels.g >> 4);
	block[2] = (color0.channels.b & 0xf0) | (color1.channels.b >> 4);
}

 void WriteColors555(__global unsigned char* block,
						   const union Color color0,
						   const union Color color1) {
	
	__constant unsigned char two_compl_trans_table[8] = {
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
	(short)((color1.channels.r >> 3) - (color0.channels.r >> 3));
	short delta_g =
	(short)((color1.channels.g >> 3) - (color0.channels.g >> 3));
	short delta_b =
	(short)((color1.channels.b >> 3) - (color0.channels.b >> 3));

	
	block[0] = (color0.channels.r & 0xf8) | two_compl_trans_table[delta_r + 4];
	block[1] = (color0.channels.g & 0xf8) | two_compl_trans_table[delta_g + 4];
	block[2] = (color0.channels.b & 0xf8) | two_compl_trans_table[delta_b + 4];
}

 void WriteCodewordTable(__global unsigned char* block,
							   unsigned char sub_block_id,
							   unsigned char table) {

	unsigned char shift = (2 + (3 - sub_block_id * 3));
	block[3] &= ~(0x07 << shift);
	block[3] |= table << shift;
}



 void WritePixelData(__global unsigned char* block, unsigned int pixel_data) {
	block[4] |= pixel_data >> 24;
	block[5] |= (pixel_data >> 16) & 0xff;
	block[6] |= (pixel_data >> 8) & 0xff;
	block[7] |= pixel_data & 0xff;
}

 void WriteFlip(__global unsigned char* block, bool flip) {
	block[3] &= ~0x01;
	block[3] |= (unsigned char)(flip);
}

 void WriteDiff(__global unsigned char* block, bool diff) {
	block[3] &= ~0x02;
	block[3] |= (unsigned char)(diff) << 1;
}

 void ExtractBlock(__global unsigned char* dst, __global const unsigned char* src, int width) {
	for (int j = 0; j < 4; ++j) {
		my_memcpy(&dst[j * 4 * 4], src, 4 * 4);
		src += width * 4;
	}
}





 union Color makeColor444(const float* bgr) {
	unsigned char b4 = round_to_4_bits(bgr[0]);
	unsigned char g4 = round_to_4_bits(bgr[1]);
	unsigned char r4 = round_to_4_bits(bgr[2]);
	union Color bgr444;
	bgr444.channels.b = (b4 << 4) | b4;
	bgr444.channels.g = (g4 << 4) | g4;
	bgr444.channels.r = (r4 << 4) | r4;
	
	bgr444.channels.a = 0x44;

	return bgr444;
}





 union Color makeColor555(const float* bgr) {
	unsigned char b5 = round_to_5_bits(bgr[0]);
	unsigned char g5 = round_to_5_bits(bgr[1]);
	unsigned char r5 = round_to_5_bits(bgr[2]);
	union Color bgr555;
	bgr555.channels.b = (b5 > 2);
	bgr555.channels.g = (g5 > 2);
	bgr555.channels.r = (r5 > 2);
	
	bgr555.channels.a = 0x55;
	return bgr555;
}

void getAverageColor(__global const union Color *src, float* avg_color)
{
	unsigned int sum_b = 0;
	unsigned int sum_g = 0;
	unsigned int sum_r = 0;

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

unsigned long computeLuminance(__global unsigned char* block,
						   __global const union Color *src,
						   const union Color base,
						   int sub_block_id,
						   const unsigned char *idx_to_num_tab,
						   unsigned long threshold)
{
	unsigned int best_tbl_err = threshold;
	unsigned char best_tbl_idx = 0;
	unsigned char best_mod_idx[8][8];  

	
	
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
				const union Color color= candidate_color[mod_idx];

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

	for (unsigned int i = 0; i < 8; ++i) {
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


bool tryCompressSolidBlock(__global unsigned char* dst,
						   __global const union Color *src,
						   unsigned long* error)
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

	unsigned char best_tbl_idx = 0;
	unsigned char best_mod_idx = 0;


	unsigned int best_mod_err = UINT32_MAX;

	
	
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		
		
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			const union Color color= makeColor(base, lum);

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

unsigned long compressBlock(__global unsigned char* dst,
							   __global const union Color *ver_src,
							   __global const union Color *hor_src,
							   unsigned long threshold)
{
	unsigned char g_idx_to_num2[4][8] = {
		{0, 4, 1, 5, 2, 6, 3, 7},        
		{8, 12, 9, 13, 10, 14, 11, 15},  
		{0, 4, 8, 12, 1, 5, 9, 13},      
		{2, 6, 10, 14, 3, 7, 11, 15}     
	};
	unsigned long solid_error = 0;
	if (tryCompressSolidBlock(dst, ver_src, &solid_error)) {


		return solid_error;
	}

	__global const union Color *sub_block_src[4] = {ver_src, ver_src + 8, hor_src, hor_src + 8};
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

	
	
	
	unsigned int sub_block_err[4] = {0};
	for (unsigned int i = 0; i < 4; ++i) {
		for (unsigned int j = 0; j < 8; ++j) {
			sub_block_err[i] += getColorError(sub_block_avg[i], sub_block_src[i][j]);
		}
	}

	bool flip = sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1];

	
	my_memset(dst, 0, 8);

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
								   g_idx_to_num2[sub_block_off_0],
								   threshold);
	
	lumi_error2 = computeLuminance(dst, sub_block_src[sub_block_off_1],
								   sub_block_avg[sub_block_off_1], 1,
								   g_idx_to_num2[sub_block_off_1],
								   threshold);

	return lumi_error1 + lumi_error2;
}

__kernel void cmprss(__global unsigned char* src,
							     __global unsigned char* dst,
								 const int width,
		                         const int height,
                                 unsigned long compressed_error) {

	__global static union Color ver_blocks[16];
	__global static union Color hor_blocks[16];

	int x = get_global_id(0);
	int y = get_global_id(1);

	__global const union Color *row0 = (__global const union Color*) (src += y * width * 4 * 4 + x * 4 * 4);
	__global const union Color *row1 = row0 + width;
	__global const union Color *row2 = row1 + width;
	__global const union Color *row3 = row2 + width;

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

	compressed_error += compressBlock(dst += y * 8 * width / 4 + x * 8, ver_blocks, hor_blocks, INT32_MAX);
}
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


void gpu_find(cl_device_id &device,
		uint platform_select,
		uint device_select,
		cl_platform_id *platform_list,
		cl_device_id *device_list)
{
	cl_platform_id platform;
	cl_uint platform_num = 0;
	platform_list = NULL;

	cl_uint device_num = 0;
	device_list = NULL;

	size_t attr_size = 0;
	cl_char* attr_data = NULL;

	
	 clGetPlatformIDs(0, NULL, &platform_num);
	platform_list = new cl_platform_id[platform_num];

	
	 clGetPlatformIDs(platform_num, platform_list, NULL);

	
	for(uint platf=0; platf<platform_num; platf++)
	{
		
		 clGetPlatformInfo(platform_list[platf],
				CL_PLATFORM_VENDOR, 0, NULL, &attr_size);
		attr_data = new cl_char[attr_size];

		
		 clGetPlatformInfo(platform_list[platf],
				CL_PLATFORM_VENDOR, attr_size, attr_data, NULL);
		delete[] attr_data;

		
		 clGetPlatformInfo(platform_list[platf],
				CL_PLATFORM_VERSION, 0, NULL, &attr_size);
		attr_data = new cl_char[attr_size];

		
		 clGetPlatformInfo(platform_list[platf],
				CL_PLATFORM_VERSION, attr_size, attr_data, NULL);
		delete[] attr_data;

		
		platform = platform_list[platf];

		
		if(clGetDeviceIDs(platform,
			CL_DEVICE_TYPE_GPU, 0, NULL, &device_num) == CL_DEVICE_NOT_FOUND) {
			device_num = 0;
			continue;
		}

		device_list = new cl_device_id[device_num];

		
		 clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU,
			  device_num, device_list, NULL);

		
		for(uint dev=0; dev<device_num; dev++)
		{
			
			 clGetDeviceInfo(device_list[dev], CL_DEVICE_NAME,
				0, NULL, &attr_size);
			attr_data = new cl_char[attr_size];

			
			 clGetDeviceInfo(device_list[dev], CL_DEVICE_NAME,
				attr_size, attr_data, NULL);
			delete[] attr_data;

			
			 clGetDeviceInfo(device_list[dev], CL_DEVICE_VERSION,
				0, NULL, &attr_size);
			attr_data = new cl_char[attr_size];

			
			 clGetDeviceInfo(device_list[dev], CL_DEVICE_VERSION,
				attr_size, attr_data, NULL);
			delete[] attr_data;

			
			if((platf == platform_select) && (dev == device_select)){
				device = device_list[dev];
			}
		}
	}

	
	
}

TextureCompressor::TextureCompressor() {
	gpu_find(this->device, 0, 0, this->platform_ids, this->device_ids);
}
TextureCompressor::~TextureCompressor() { }	

unsigned long TextureCompressor::compress(const uint8_t* src,
									  uint8_t* dst,
									  int width,
									  int height)
{
	int				sizeSrc, sizeDst;
	size_t			global[2];
	string			kernel_src;
	cl_context		context;
	cl_command_queue commands;
	cl_program		program;
	cl_kernel		kernel;
	cl_uint			nd;
	cl_mem			src_in;
	cl_mem			dst_out;
	cl_mem			c_err;
	int				i, ret;

	unsigned long compressed_error = 0;
	cl_device_id device_id = this->device;

	
	
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &ret);
	CL_ERR(ret);

	commands = clCreateCommandQueue(context, device_id,
									CL_QUEUE_PROFILING_ENABLE, &ret);
	CL_ERR(ret);
	
	sizeSrc = width * height * 4;
	sizeDst = width * height * 4 / 8;

	
	
	src_in   = clCreateBuffer(context,  CL_MEM_READ_ONLY,
							sizeSrc, NULL, NULL);
	c_err = clCreateBuffer(context,  CL_MEM_WRITE_ONLY,
							sizeof(unsigned long), NULL, NULL);
	dst_out  = clCreateBuffer(context,  CL_MEM_WRITE_ONLY,
							sizeDst, NULL, NULL);

	
	read_kernel("sorin_kernel.cl", kernel_src);
	const char* kernel_c_str = kernel_src.c_str();

	
	program = clCreateProgramWithSource(context, 1,
				(const char **) &kernel_c_str, NULL, &ret);
	CL_ERR(ret);
	
	ret = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    CL_COMPILE_ERR(ret, program, device_id);
	
	kernel = clCreateKernel(program, "cmprss", &ret);
	CL_ERR(ret);
	
	ret  = 0;
	ret  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &src_in);
	ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &dst_out);
    ret |= clSetKernelArg(kernel, 2, sizeof(int), &width);
    ret |= clSetKernelArg(kernel, 3, sizeof(int), &height);
    ret |= clSetKernelArg(kernel, 4, sizeof(unsigned long), &c_err);
	
	ret = clEnqueueWriteBuffer(commands, src_in, CL_TRUE, 0,
							   sizeSrc, src, 0, NULL, NULL);

	cl_event prof_event;
	global[0] =(size_t) width / 4;
	global[1] =(size_t) height / 4;

	ret = clEnqueueNDRangeKernel(commands, kernel, 2, NULL,
								 global, NULL, 0, NULL, &prof_event); 
	
	clFinish(commands);
	cl_ulong run_time = (cl_ulong)0;
	cl_ulong ev_start_time = (cl_ulong)0;
	cl_ulong ev_end_time = (cl_ulong)0;

	
	ret = clEnqueueReadBuffer( commands, dst_out, CL_TRUE, 0,
							  sizeDst, dst, 0, NULL, NULL );


	ret = clEnqueueReadBuffer( commands, c_err, CL_TRUE, 0,
							  sizeof(unsigned long), &compressed_error, 0, NULL, NULL);

	
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseMemObject(src_in);
	clReleaseMemObject(c_err);
	clReleaseMemObject(dst_out);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);

	return compressed_error;
}
