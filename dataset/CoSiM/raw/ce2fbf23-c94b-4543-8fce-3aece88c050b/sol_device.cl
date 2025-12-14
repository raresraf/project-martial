
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

uchar round_to_5_bits(float val) {
	return clamp(val * 31.0f / 255.0f + 0.5f, 0.0f, 31.0f);
}

uchar round_to_4_bits(float val) {
	return clamp(val * 15.0f / 255.0f + 0.5f, 0.0f, 15.0f);
}

void * my_memcpy (void *dest, const void *src, size_t len)
{
  char *d = dest;
  const char *s = src;
  while (len--)
    *d++ = *s++;
	return dest;
}

void * my_memset(void *b, int c, int len)
{
  int i;
  unsigned char *p = b;
  i = 0;
  while(len > 0)
    {
      *p = c;
      p++;
      len--;
    }
  return b;
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

__constant uchar g_mod_to_pix[4] = {3, 2, 0, 1};

__constant uchar g_idx_to_num[4][8] = {
	{0, 4, 1, 5, 2, 6, 3, 7},        
	{8, 12, 9, 13, 10, 14, 11, 15},  


	{0, 4, 8, 12, 1, 5, 9, 13},      
	{2, 6, 10, 14, 3, 7, 11, 15}     
};

union Color makeColor(const union Color* base, short lum) {
	int b = (int)(base->channels.b) + lum;
	int g = (int)(base->channels.g) + lum;
	int r = (int)(base->channels.r) + lum;
	union Color color;
	color.channels.b = (uchar)(clamp(b, 0, 255));
	color.channels.g = (uchar)(clamp(g, 0, 255));
	color.channels.r = (uchar)(clamp(r, 0, 255));
	return color;
}

uint getColorError(const union Color* u, const union Color* v) {
#ifdef USE_PERCEIVED_ERROR_METRIC
	float delta_b = (float)(u->channels.b) - v->channels.b;
	float delta_g = (float)(u->channels.g) - v->channels.g;
	float delta_r = (float)(u->channels.r) - v->channels.r;
	return (uint)(0.299f * delta_b * delta_b +
								 0.587f * delta_g * delta_g +
								 0.114f * delta_r * delta_r);
#else
	int delta_b = (int)(u->channels.b) - v->channels.b;
	int delta_g = (int)(u->channels.g) - v->channels.g;
	int delta_r = (int)(u->channels.r) - v->channels.r;
	return delta_b * delta_b + delta_g * delta_g + delta_r * delta_r;
#endif
}

void WriteColors444(uchar* block,
						   const union Color* color0,
						   const union Color* color1) {
	block[0] = (color0->channels.r & 0xf0) | (color1->channels.r >> 4);
	block[1] = (color0->channels.g & 0xf0) | (color1->channels.g >> 4);
	block[2] = (color0->channels.b & 0xf0) | (color1->channels.b >> 4);
}

void WriteColors555(uchar* block,
						   const union Color* color0,
						   const union Color* color1) {
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
	(short)(color1->channels.r >> 3) - (color0->channels.r >> 3);
	short delta_g =
	(short)(color1->channels.g >> 3) - (color0->channels.g >> 3);
	short delta_b =
	(short)(color1->channels.b >> 3) - (color0->channels.b >> 3);

	block[0] = (color0->channels.r & 0xf8) | two_compl_trans_table[delta_r + 4];
	block[1] = (color0->channels.g & 0xf8) | two_compl_trans_table[delta_g + 4];
	block[2] = (color0->channels.b & 0xf8) | two_compl_trans_table[delta_b + 4];
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

void WriteFlip(uchar* block, bool flip) {
	block[3] &= ~0x01;
	block[3] |= (uchar)(flip);
}



void WriteDiff(uchar* block, bool diff) {
	block[3] &= ~0x02;
	block[3] |= (uchar)(diff) << 1;
}

void ExtractBlock(uchar* dst, const uchar* src, int width) {
	for (int j = 0; j < 4; ++j) {


		my_memcpy(&dst[j * 4 * 4], src, 4 * 4);
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

unsigned long computeLuminance(uchar* block,
						   const union Color* src,
						   const union Color* base,
						   int sub_block_id,
						   const uchar idx_to_num_tab,
						   unsigned long threshold)
{
	uint best_tbl_err = threshold;
	uchar best_tbl_idx = 0;
	uchar best_mod_idx[8][8];  

	
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		
		
		union Color candidate_color[4];  
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			candidate_color[mod_idx] = makeColor(base, lum);
		}

		uint tbl_err = 0;



		for (unsigned int i = 0; i < 8; ++i) {
			
			
			uint best_mod_err = threshold;
			for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
				const union Color color = candidate_color[mod_idx];

				uint mod_err = getColorError(&src[i], &color);
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

		
		int texel_num = g_idx_to_num[idx_to_num_tab][i];
		pix_data |= msb << (texel_num + 16);
		pix_data |= lsb << (texel_num);
	}

	WritePixelData(block, pix_data);

	return best_tbl_err;
}

bool tryCompressSolidBlock(uchar* dst,
						   const union Color* src,
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
	WriteColors555(dst, &base, &base);

	uchar best_tbl_idx = 0;
	uchar best_mod_idx = 0;
	uint best_mod_err = UINT_MAX;

	
	
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		
		
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			const union Color color = makeColor(&base, lum);

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
	return true;
}




unsigned long compressBlock(uchar* dst,
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
			sub_block_err[i] += getColorError(&sub_block_avg[i], &sub_block_src[i][j]);
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
		WriteColors555(dst, &sub_block_avg[sub_block_off_0],
					   &sub_block_avg[sub_block_off_1]);
	} else {
		WriteColors444(dst, &sub_block_avg[sub_block_off_0],
					   &sub_block_avg[sub_block_off_1]);
	}

	unsigned long lumi_error1 = 0, lumi_error2 = 0;
	
	lumi_error1 = computeLuminance(dst, sub_block_src[sub_block_off_0],
								   &sub_block_avg[sub_block_off_0], 0,
								   sub_block_off_0,
								   threshold);
	
	lumi_error2 = computeLuminance(dst, sub_block_src[sub_block_off_1],
								   &sub_block_avg[sub_block_off_1], 1,
								   sub_block_off_1,
								   threshold);

	return lumi_error1 + lumi_error2;
}

__kernel void compress(__global uchar * src,
											 __global uchar * dst,
											 const int width,
											 const int height,
											 __global unsigned long * err)
{
	union Color ver_blocks[16];
	union Color hor_blocks[16];
	unsigned long compressed_error = 0;

	
	int y = get_global_id(0);
	int x = get_global_id(1);

	
	const union Color* row0 = src + (y * width * 16 + x * 16);
	const union Color* row1 = row0 + width;
	const union Color* row2 = row1 + width;
	const union Color* row3 = row2 + width;

	
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

	uchar loc_dst[8];

	
	compressed_error += compressBlock(loc_dst, ver_blocks, hor_blocks, INT_MAX);
	*err += compressed_error;

	for(int i = 0; i < 8; i++)
	{
		dst[y * width * 2 + x * 8 + i] = loc_dst[i];
	}
}
#include "compress.hpp"

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


int CL_ERR(int cl_ret)
{
	if(cl_ret != CL_SUCCESS){
		cout << endl << cl_get_string_err(cl_ret) << endl;
		return 1;
	}
	return 0;
}

#define DIE(assertion, call_description)  \
do { \
	if (assertion) { \
		fprintf(stderr, "(%d): ", __LINE__); \
		perror(call_description); \
		exit(EXIT_FAILURE); \
	} \
} while(0);


void gpu_find(cl_device_id &device,
		uint device_select)
{
	cl_platform_id platform;
	cl_uint platform_num = 0;
	cl_platform_id* platform_list = NULL;

	cl_uint device_num = 0;
	cl_device_id* device_list = NULL;
	cl_uint bec = 0;

	
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

		for(uint dev=0; dev<device_num; dev++)
		{

			
			if(dev == device_select){
				device = device_list[dev];
				bec = 1;
				break;
			}
		}
		if(bec)
			break;
	}
  
	delete[] platform_list;
	delete[] device_list;
}

TextureCompressor::TextureCompressor() {
		gpu_find(this->device, 0);
		DIE(this->device == 0, "check valid device");
 } 	
TextureCompressor::~TextureCompressor() { }	


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


void read_kernel(string file_name, string &str_kernel)
{
	ifstream in_file(file_name.c_str());
	in_file.open(file_name.c_str());
	DIE( !in_file.is_open(), "ERR OpenCL kernel file. Same directory as binary ?" );

	stringstream str_stream;
	str_stream << in_file.rdbuf();

	str_kernel = str_stream.str();
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


unsigned long gpu_execute_kernel(cl_device_id device,
			const char* kernel_name,
			uint num_times,
			const uint8_t* src,
			uint8_t* dst,
			int width,
			int height)
{
	cl_int ret;
	uint size_src;
	cl_context context;
	cl_command_queue command_queue;
	cl_program program;
	cl_kernel kernel;
	cl_mem mat_in;
	cl_mem mat_out;
  cl_mem err_out;
  unsigned long err = 0;

	string kernel_src;
	
	context = clCreateContext(0, 1, &device, NULL, NULL, &ret);
	CL_ERR( ret );

  
	command_queue = clCreateCommandQueue(context, device, 0, &ret);
	CL_ERR( ret );

  size_src = width * height;

  
  mat_in = clCreateBuffer(context,  CL_MEM_READ_ONLY,
							sizeof(uint8_t) * size_src * 4, NULL, NULL);
	mat_out = clCreateBuffer(context,  CL_MEM_READ_WRITE,
							sizeof(uint8_t) * size_src * 4 / 8, NULL, NULL);
  err_out = clCreateBuffer(context, CL_MEM_READ_WRITE,
              sizeof(unsigned long), NULL, NULL);

	
	read_kernel("sol_device.cl", kernel_src);
	const char* kernel_c_str = kernel_src.c_str();

	
	program = clCreateProgramWithSource(context, 1,
		  &kernel_c_str, NULL, &ret);
	CL_ERR( ret );

	
	ret = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	CL_COMPILE_ERR( ret, program, device );

	
	kernel = clCreateKernel(program, kernel_name, &ret);
	CL_ERR( ret );

	
  CL_ERR(clSetKernelArg(kernel, 0, sizeof(cl_mem), &mat_in));
	CL_ERR(clSetKernelArg(kernel, 1, sizeof(cl_mem), &mat_out));
	CL_ERR(clSetKernelArg(kernel, 2, sizeof(int), &width));
	CL_ERR(clSetKernelArg(kernel, 3, sizeof(int), &height));
  CL_ERR(clSetKernelArg(kernel, 4, sizeof(cl_mem), &err_out));

  
	CL_ERR(clEnqueueWriteBuffer(command_queue, mat_in, CL_TRUE, 0,
							   sizeof(uint8_t) * size_src * 4, src, 0, NULL, NULL));
  CL_ERR(clEnqueueWriteBuffer(command_queue, err_out, CL_TRUE, 0,
                 sizeof(unsigned long), &err, 0, NULL, NULL));

  
	size_t globalSize[2] = {(size_t)height/4, (size_t)width/4};
	for(uint i=0; i<num_times; i++){
		ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL,
			globalSize, NULL, 0, NULL, NULL);
		CL_ERR( ret );
	}

	
	CL_ERR( clFinish(command_queue) );

	
	CL_ERR(clEnqueueReadBuffer(command_queue, mat_out, CL_TRUE, 0,
									sizeof(uint8_t) * size_src * 4 / 8, dst, 0, NULL, NULL ));
  CL_ERR(clEnqueueReadBuffer(command_queue, err_out, CL_TRUE, 0,
                	sizeof(unsigned long), &err, 0, NULL, NULL ));
	
	CL_ERR( clReleaseMemObject(mat_in) );
	CL_ERR( clReleaseMemObject(mat_out) );
  CL_ERR( clReleaseMemObject(err_out) );
	CL_ERR( clReleaseCommandQueue(command_queue) );
	CL_ERR( clReleaseContext(context) );
  return err;
}

unsigned long TextureCompressor::compress(const uint8_t* src,
									  uint8_t* dst,
									  int width,
									  int height)
{
	return gpu_execute_kernel(this->device, "compress", 1, src, dst, width, height);
}
