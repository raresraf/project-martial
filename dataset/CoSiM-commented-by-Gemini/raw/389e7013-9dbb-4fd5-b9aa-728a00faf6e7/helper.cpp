/**
 * @file helper.cpp
 * @brief OpenCL host utility services.
 * Functional Utility: Centralizes error diagnostic logic and kernel management 
 * for the host application, enabling robust heterogeneous compute execution.
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <CL/cl.h>

#include "helper.hpp"

using namespace std;

/**
 * @brief Validates OpenCL API return codes.
 * @param cl_ret return status from an OpenCL function.
 * @return 0 on success, 1 on failure.
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
 * @brief Validates OpenCL build status and retrieves logs for troubleshooting.
 */
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

/**
 * @brief Reads OpenCL kernel source from an external file.
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
 * @brief Maps OpenCL error codes to human-readable strings.
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
 * @brief Retrieves build log for specific hardware diagnostics.
 */
void cl_get_compiler_err_log(cl_program program,
                             cl_device_id device)
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

/**
 * @file texture_compress.cl
 * @brief OpenCL kernel logic for Ericsson Texture Compression (ETC1).
 * Architectural Intent: Parallel implementation of the ETC1 standard, 
 * optimized for GPU SIMD execution and efficient memory access patterns.
 */

/*
#define UINT32_MAX  (0xffffffff)
#define INT32_MAX   (2147483647)

// @struct BgraColorType
// @brief Raw channel mapping for texture texels.
typedef struct BgraColorType {
    uchar b;
    uchar g;
    uchar r;
    uchar a;
} BgraColorType;

// @struct Color
// @brief Unified representation for color manipulation.
typedef union Color {
    BgraColorType channels;
    uchar components[4];
	uint bits;
} Color;

// clamp2: restricted range utility.
uchar clamp2( uchar val,  uchar min,  uchar max) {
	return (uchar)(val < min ? min : (val > max ? max : val));
}

// round_to_5_bits: Quantizes channel to 5-bit depth.
uchar round_to_5_bits( float val) {
	return clamp2((uchar)(val * 31.0f / 255.0f + 0.5f), 0, 31);
}

// round_to_4_bits: Quantizes channel to 4-bit depth.
uchar round_to_4_bits( float val) {
	return clamp2((uchar)(val * 15.0f / 255.0f + 0.5f), 0, 15);
}

// Standard ETC1 codeword tables.
__attribute__((aligned(16))) __constant short g_codeword_tables[8][4] = {
	{-8, -2, 2, 8},
	{-17, -5, 5, 17},
	{-29, -9, 9, 29},
	{-42, -13, 13, 42},
	{-60, -18, 18, 60},
	{-80, -24, 24, 80},
	{-106, -33, 33, 106},
	{-183, -47, 47, 183}};

// makeColor: Applies luminance modifier to base color.
Color makeColor(const Color base, short lum) {
	int b = (int)(base.channels.b) + lum;
	int g = (int)(base.channels.g) + lum;
	int r = (int)(base.channels.r) + lum;
	Color color;
	color.channels.b = (uchar)(clamp2(b, 0, 255));
	color.channels.g = (uchar)(clamp2(g, 0, 255));
	color.channels.r = (uchar)(clamp2(r, 0, 255));
	return color;
}

// getColorError: Evaluates perceptual distance between color points.
uint getColorError(const Color u, const Color v) {
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
	return delta_b * delta_b + 
           delta_g * delta_g + 
           delta_r * delta_r;
#endif
}

// WriteColors444: Serializes Individual mode blocks.
void WriteColors444(uchar* block,
					const Color color0,
					const Color color1) {
	block[0] = (color0.channels.r & 0xf0) | (color1.channels.r >> 4);
	block[1] = (color0.channels.g & 0xf0) | (color1.channels.g >> 4);
	block[2] = (color0.channels.b & 0xf0) | (color1.channels.b >> 4);
}

// WriteColors555: Serializes Differential mode blocks.
void WriteColors555(uchar* block,
					const Color color0,
					const Color color1) {
	const uchar two_compl_trans_table[8] = {
		4, 5, 6, 7, 0, 1, 2, 3,
	};
	
	short delta_r = (short)(color1.channels.r >> 3) - (color0.channels.r >> 3);
	short delta_g = (short)(color1.channels.g >> 3) - (color0.channels.g >> 3);
	short delta_b = (short)(color1.channels.b >> 3) - (color0.channels.b >> 3);
	
	block[0] = (color0.channels.r & 0xf8) | two_compl_trans_table[delta_r + 4];
	block[1] = (color0.channels.g & 0xf8) | two_compl_trans_table[delta_g + 4];
	block[2] = (color0.channels.b & 0xf8) | two_compl_trans_table[delta_b + 4];
}

// WriteCodewordTable: Encodes table index.
void WriteCodewordTable( uchar* block,  uchar sub_block_id,  uchar table) {
	uchar shift = (2 + (3 - sub_block_id * 3));
	block[3] &= ~(0x07 << shift);
	block[3] |= table << shift;
}

// WritePixelData: Encodes pixel-level bitstream.
void WritePixelData( uchar* block,  uint pixel_data) {
	block[4] |= pixel_data >> 24;
	block[5] |= (pixel_data >> 16) & 0xff;
	block[6] |= (pixel_data >> 8) & 0xff;
	block[7] |= pixel_data & 0xff;
}

// WriteFlip: Sets orientation bit.
void WriteFlip( uchar* block,  bool flip) {
	block[3] &= ~0x01;
	block[3] |= (uchar)(flip);
}

// WriteDiff: Sets color mode bit.
void WriteDiff( uchar* block,  bool diff) {
	block[3] &= ~0x02;
	block[3] |= (uchar)(diff) << 1;
}

// individual mode quantization.
Color makeColor444( const float* bgr) {
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

// differential mode quantization.
Color makeColor555( const float* bgr) {
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

// getAverageColor: Energy center estimation.
void getAverageColor( const Color* src,  float* avg_color)
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

// computeLuminance: Optimal search across tables.
ulong computeLuminance(uchar* block,
                       const Color* src,
					   const Color base,
					   int sub_block_id,
					   const uchar* idx_to_num_tab,
					   ulong threshold)
{
	uint best_tbl_err = threshold;
	uchar best_tbl_idx = 0;
	uchar best_mod_idx[8][8];  
    uchar g_mod_to_pix[4] = {3, 2, 0, 1};

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
				const Color color = candidate_color[mod_idx];
				uint mod_err = getColorError(src[i], color);
				if (mod_err < best_mod_err) {
					best_mod_idx[tbl_idx][i] = mod_idx;
					best_mod_err = mod_err;
					if (mod_err == 0) break;  
				}
			}
			tbl_err += best_mod_err;
			if (tbl_err > best_tbl_err) break;  
		}
		if (tbl_err < best_tbl_err) {
			best_tbl_err = tbl_err;
			best_tbl_idx = tbl_idx;
			if (tbl_err == 0) break;  
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

// tryCompressSolidBlock: Performance path for uniform colors.
bool tryCompressSolidBlock(uchar* dst,
						   const Color* src,
						   unsigned long* error)
{
	for (unsigned int i = 1; i < 16; ++i) {
		if (src[i].bits != src[0].bits)
			return false;
	}
    for (unsigned int i = 0; i < 8; i++)
        dst[i] = 0;
	
	float src_color_float[3] = {(float)(src->channels.b), (float)(src->channels.g), (float)(src->channels.r)};
	Color base = makeColor555(src_color_float);
	
	WriteDiff(dst, true);
	WriteFlip(dst, false);
	WriteColors555(dst, base, base);
	
	uchar best_tbl_idx = 0;
	uchar best_mod_idx = 0;
	uint best_mod_err = UINT32_MAX; 
	uchar g_mod_to_pix[4] = {3, 2, 0, 1};
    uchar g_idx_to_num[4][8] = {
        {0, 4, 1, 5, 2, 6, 3, 7}, {8, 12, 9, 13, 10, 14, 11, 15},
        {0, 4, 8, 12, 1, 5, 9, 13}, {2, 6, 10, 14, 3, 7, 11, 15}
    };

	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			const Color color = makeColor(base, lum);
			uint mod_err = getColorError(*src, color);
			if (mod_err < best_mod_err) {
				best_tbl_idx = tbl_idx;
				best_mod_idx = mod_idx;
				best_mod_err = mod_err;
				if (mod_err == 0) break;  
			}
		}
		if (best_mod_err == 0) break;
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

// compressBlock: Full structural logic for encoding 4x4 blocks.
unsigned long compressBlock(uchar* dst,
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
			if (component_diff < -4 || component_diff > 3) {
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
	bool flip = sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1];
    for (unsigned int i = 0; i < 8; i++)
        dst[i] = 0;
    WriteDiff(dst, use_differential[!!flip]);
	WriteFlip(dst, flip);
	uchar sub_block_off_0 = flip ? 2 : 0;
	uchar sub_block_off_1 = sub_block_off_0 + 1;
	if (use_differential[!!flip]) {
		WriteColors555(dst, sub_block_avg[sub_block_off_0], sub_block_avg[sub_block_off_1]);
	} else {
		WriteColors444(dst, sub_block_avg[sub_block_off_0], sub_block_avg[sub_block_off_1]);
	}
	unsigned long lumi_error1 = 0, lumi_error2 = 0;
	uchar g_idx_to_num[4][8] = {
        {0, 4, 1, 5, 2, 6, 3, 7}, {8, 12, 9, 13, 10, 14, 11, 15},
        {0, 4, 8, 12, 1, 5, 9, 13}, {2, 6, 10, 14, 3, 7, 11, 15}
    };
	lumi_error1 = computeLuminance(dst, sub_block_src[sub_block_off_0], sub_block_avg[sub_block_off_0], 0, g_idx_to_num[sub_block_off_0], threshold);
	lumi_error2 = computeLuminance(dst, sub_block_src[sub_block_off_1], sub_block_avg[sub_block_off_1], 1, g_idx_to_num[sub_block_off_1], threshold);
	return lumi_error1 + lumi_error2;
}

void myMemCpy(void *dest, void *src, size_t n)
{
   char *csrc = (char *)src;
   char *cdest = (char *)dest;
   for (int i=0; i<n; i++)
       cdest[i] = csrc[i];
}

// compress: OpenCL kernel for massively parallel texture compression.
__kernel void compress(__global uchar* src, __global uchar* dst, int width, int height)
{
	uint block_row_index = get_global_id(0);
	uint block_col_index = get_global_id(1);
    Color ver_blocks[16];
	Color hor_blocks[16];
	unsigned long compressed_error = 0;
    int y = block_row_index * 4;
    src += y * width * 4 * 4;
    int x = block_col_index * 4;
    dst += x * 8;
	Color* row0 = (Color*)(__private)(src + x * 4);
	Color* row1 = row0 + width;
	Color* row2 = row1 + width;
	Color* row3 = row2 + width;
	myMemCpy(ver_blocks, row0, 8);
	myMemCpy(ver_blocks + 2, row1, 8);
	myMemCpy(ver_blocks + 4, row2, 8);
	myMemCpy(ver_blocks + 6, row3, 8);
	myMemCpy(ver_blocks + 8, row0 + 2, 8);
	myMemCpy(ver_blocks + 10, row1 + 2, 8);
	myMemCpy(ver_blocks + 12, row2 + 2, 8);
	myMemCpy(ver_blocks + 14, row3 + 2, 8);
	myMemCpy(hor_blocks, row0, 16);
	myMemCpy(hor_blocks + 4, row1, 16);
	myMemCpy(hor_blocks + 8, row2, 16);
	myMemCpy(hor_blocks + 12, row3, 16);
	compressed_error += compressBlock((__private)dst, ver_blocks, hor_blocks, INT32_MAX);
}
*/

/**
 * @file texture_compress_skl.cpp
 * @brief Host management orchestration for OpenCL texture compression.
 */

/*
#include "compress.hpp"
#include <iostream>
#include <vector>
#include <CL/cl.h>
#include "helper.hpp"

using namespace std;

// gpu_find: Automatically discovers and selects compute device.
void gpu_find(cl_device_id &device)
{
	cl_uint platform_num = 0;
	cl_platform_id* platform_list = NULL;
	cl_uint device_num = 0;
	cl_device_id* device_list = NULL;
	CL_ERR( clGetPlatformIDs(0, NULL, &platform_num));
	platform_list = new cl_platform_id[platform_num];
	CL_ERR( clGetPlatformIDs(platform_num, platform_list, NULL));
	for(uint platf = 0; platf < platform_num; platf++)
	{
		cl_platform_id platform = platform_list[platf];
		if(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &device_num) == CL_DEVICE_NOT_FOUND) {
			continue;
		}
		device_list = new cl_device_id[device_num];
		CL_ERR( clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, device_num, device_list, NULL));
		device = device_list[0];
		delete[] device_list;
		break;
	}
	delete[] platform_list;
}

TextureCompressor::TextureCompressor() {gpu_find(device);} 
TextureCompressor::~TextureCompressor() { }
	
unsigned long TextureCompressor::compress(const uint8_t* src,
									  uint8_t* dst,
									  int width,
									  int height)
{
	cl_int ret;
	cl_context context;
	cl_command_queue cmd_queue;
	context = clCreateContext(0, 1, &device, NULL, NULL, &ret);
	CL_ERR( ret );
	cmd_queue = clCreateCommandQueue(context, device, 0, &ret);
	CL_ERR( ret );
	cl_mem buf_src = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_uchar) * width * height * 4, NULL, &ret);
	CL_ERR( ret );
	cl_mem buf_dst = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_uchar) * width * height * 4 / 8, NULL, &ret);
	CL_ERR( ret );
	CL_ERR( clEnqueueWriteBuffer(cmd_queue, buf_src, CL_TRUE, 0, sizeof(cl_uchar) * width * height * 4, src, 0, NULL, NULL));
	string kernel_src;
	read_kernel("texture_compress.cl", kernel_src);
	const char* kernel_c_str = kernel_src.c_str();
	cl_program program = clCreateProgramWithSource(context, 1, &kernel_c_str, NULL, &ret);
	CL_ERR( ret );
	ret = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	CL_COMPILE_ERR( ret, program, device );
	cl_kernel kernel = clCreateKernel(program, "compress", &ret);
	CL_ERR( ret );
	CL_ERR( clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&buf_src) );
	CL_ERR( clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&buf_dst) );
	CL_ERR( clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&width) );
	CL_ERR( clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&height) );
	cl_event event;
	size_t globalSize[2] = {(size_t)(height / 4) - 1, (size_t)(width / 4) - 1};
	ret = clEnqueueNDRangeKernel(cmd_queue, kernel, 2, NULL, globalSize, 0, 0, NULL, &event);
	CL_ERR( ret );
	CL_ERR( clWaitForEvents(1, &event));
	CL_ERR( clEnqueueReadBuffer(cmd_queue, buf_dst, CL_TRUE, 0, sizeof(cl_uchar) * width * height * 4 / 8, dst, 0, NULL, NULL) );
	CL_ERR( clFinish(cmd_queue) );
	clReleaseMemObject(buf_src);
	clReleaseMemObject(buf_dst);
	clReleaseCommandQueue(cmd_queue);
	clReleaseContext(context);
	return 0;
}
*/
