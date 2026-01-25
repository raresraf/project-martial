
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
>>>> file: skl_device.cl
#define INT32_MAX (2147483647L)
#define UINT32_MAX (4294967295UL)

typedef union {
	struct BgraColorType {
		uchar b;
		uchar g;
		uchar r;
		uchar a;
	} channels;
	uchar components[4];
	int bits;
} Color;


#define memcpy(dst, src, size) \
	{ \
	uchar *dst2 = (uchar*)dst; \
	uchar *src2 = (uchar*)src; \
	for (int i = 0; i < size; i++) \
		dst2[i] = src2[i]; \
	}\

uchar clampUC(uchar val, uchar min, uchar max) {
	return val  max ? max : val);
}

int clampInt(int val, int min, int max) {
	return val  max ? max : val);
}

float clampFl(float val, float min, float max) {
	return val  max ? max : val);
}

uchar round_to_4_bits(float val) {
	return (uchar) clampFl(val * 15.0f / 255.0f + 0.5f, 0, 15);
}

uchar round_to_5_bits(float val) {
	return (uchar) clampFl(val * 31.0f / 255.0f + 0.5f, 0, 31);
}

void memset(uchar *src, uchar c, int size)
{
	for (int i = 0; i < size; i++)
		src[i] = c;
}

Color makeColor(const Color base, short lum) {
	int b = (int)(base.channels.b) + lum;
	int g = (int)(base.channels.g) + lum;
	int r = (int)(base.channels.r) + lum;
	Color color;
	color.channels.b = (uchar)(clampInt(b, 0, 255));
	color.channels.g = (uchar)(clampInt(g, 0, 255));
	color.channels.r = (uchar)(clampInt(r, 0, 255));

	return color;
}

// Calculates the error metric for two colors. A small error signals that the
// colors are similar to each other, a large error the signals the opposite.
int getColorError(const Color u, const Color v) {
#ifdef USE_PERCEIVED_ERROR_METRIC
	float delta_b = (float)(u.channels.b) - v.channels.b;
	float delta_g = (float)(u.channels.g) - v.channels.g;
	float delta_r = (float)(u.channels.r) - v.channels.r;

	return (int)(	0.299f * delta_b * delta_b +
			0.587f * delta_g * delta_g +
			0.114f * delta_r * delta_r);
#else
	int delta_b = (int)(u.channels.b) - v.channels.b;
	int delta_g = (int)(u.channels.g) - v.channels.g;
	int delta_r = (int)(u.channels.r) - v.channels.r;

	return delta_b * delta_b + delta_g * delta_g + delta_r * delta_r;
#endif
}

Color makeColor444(const float* bgr) {
	uchar b4 = round_to_4_bits(bgr[0]);
	uchar g4 = round_to_4_bits(bgr[1]);
	uchar r4 = round_to_4_bits(bgr[2]);
	Color bgr444;
	bgr444.channels.b = (b4 << 4) | b4;
	bgr444.channels.g = (g4 << 4) | g4;
	bgr444.channels.r = (r4 << 4) | r4;
	// Added to distinguish between expanded 555 and 444 colors.
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
	// Added to distinguish between expanded 555 and 444 colors.
	bgr555.channels.a = 0x55;
	return bgr555;
}

void WriteColors444(uchar *block, const Color color0, const Color color1) {
	block[0] = (color0.channels.r & 0xf0) | (color1.channels.r >> 4);
	block[1] = (color0.channels.g & 0xf0) | (color1.channels.g >> 4);
	block[2] = (color0.channels.b & 0xf0) | (color1.channels.b >> 4);
}

void WriteColors555(uchar *block, const Color color0, const Color color1) {
	uchar two_compl_trans_table[8] = {
		4,  // -4 (100b)
		5,  // -3 (101b)
		6,  // -2 (110b)
		7,  // -1 (111b)
		0,  //  0 (000b)
		1,  //  1 (001b)
		2,  //  2 (010b)
		3,  //  3 (011b)
	};

	short delta_r = (short)((color1.channels.r >> 3) - (color0.channels.r >> 3));
	short delta_g = (short)((color1.channels.g >> 3) - (color0.channels.g >> 3));
	short delta_b = (short)((color1.channels.b >> 3) - (color0.channels.b >> 3));

	block[0] = (color0.channels.r & 0xf8) | two_compl_trans_table[delta_r + 4];
	block[1] = (color0.channels.g & 0xf8) | two_compl_trans_table[delta_g + 4];
	block[2] = (color0.channels.b & 0xf8) | two_compl_trans_table[delta_b + 4];
}

void WriteCodewordTable(uchar* block, uchar sub_block_id, uchar table) {
	uchar shift = (2 + (3 - sub_block_id * 3));
	block[3] &= ~(0x07 << shift);
	block[3] |= table << shift;
}

void WritePixelData(uchar* block, int pixel_data) {
	block[4] |= pixel_data >> 24;
	block[5] |= (pixel_data >> 16) & 0xff;
	block[6] |= (pixel_data >> 8) & 0xff;
	block[7] |= pixel_data & 0xff;
}



void getAverageColor(const Color* src, float* avg_color)
{
	uint sum_b = 0, sum_g = 0, sum_r = 0;

	for (uint i = 0; i < 8; ++i) {
		sum_b += src[i].channels.b;
		sum_g += src[i].channels.g;
		sum_r += src[i].channels.r;
	}

	__const float kInv8 = 1.0f / 8.0f;
	avg_color[0] = (float)(sum_b) * kInv8;
	avg_color[1] = (float)(sum_g) * kInv8;
	avg_color[2] = (float)(sum_r) * kInv8;
}

ulong computeLuminance(uchar* block, const Color* src, const Color base, int sub_block_id, const uchar* idx_to_num_tab, ulong threshold) {
	uint best_tbl_err = threshold;
	uchar best_tbl_idx = 0;
	uchar best_mod_idx[8][8];  // [table][texel]

	__private short codeword_tables[8][4] = {
		{ -8, -2, 2, 8 },
		{ -17, -5, 5, 17 },
		{ -29, -9, 9, 29 },
		{ -42, -13, 13, 42 },
		{ -60, -18, 18, 60 },
		{ -80, -24, 24, 80 },
		{ -106, -33, 33, 106 },
		{ -183, -47, 47, 183 } };

	// Try all codeword tables to find the one giving the best results for this
	// block.
	for (uint tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		// Pre-compute all the candidate colors; combinations of the base color and
		// all available luminance values.


		Color candidate_color[4];  // [modifier]
		for (uint mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = codeword_tables[tbl_idx][mod_idx];
			candidate_color[mod_idx] = makeColor(base, lum);


		}
		uint tbl_err = 0;
		for (uint i = 0; i < 8; ++i) {
			// Try all modifiers in the current table to find which one gives the
			// smallest error.
			uint best_mod_err = threshold;
			for (uint mod_idx = 0; mod_idx < 4; ++mod_idx) {
				const Color color = candidate_color[mod_idx];

				uint mod_err = getColorError(src[i], color);
				if (mod_err < best_mod_err) {
					best_mod_idx[tbl_idx][i] = mod_idx;
					best_mod_err = mod_err;

					if (mod_err == 0)
						break;  // We cannot do any better than this.
				}
			}

			tbl_err += best_mod_err;
			if (tbl_err > best_tbl_err)
				break;  // We're already doing worse than the best table so skip.
		}

		if (tbl_err < best_tbl_err) {
			best_tbl_err = tbl_err;
			best_tbl_idx = tbl_idx;

			if (tbl_err == 0)
				break;  // We cannot do any better than this.
		}
	}

	WriteCodewordTable(block, sub_block_id, best_tbl_idx);

	int pix_data = 0;

	__const uchar g_mod_to_pix[4] = { 3, 2, 0, 1 };

	for (uint i = 0; i < 8; ++i) {
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

	return best_tbl_err;
}


bool tryCompressSolidBlock(uchar* dst,
		const Color* src, ulong *error) {
	
	__private short codeword_tables[8][4] = {
		{ -8, -2, 2, 8 },
		{ -17, -5, 5, 17 },
		{ -29, -9, 9, 29 },
		{ -42, -13, 13, 42 },
		{ -60, -18, 18, 60 },
		{ -80, -24, 24, 80 },
		{ -106, -33, 33, 106 },
		{ -183, -47, 47, 183 } };

	__const uchar g_mod_to_pix[4] = { 3, 2, 0, 1 };

	__const uchar g_idx_to_num[4][8] = {
		{ 0, 4, 1, 5, 2, 6, 3, 7 },        // Vertical block 0.
		{ 8, 12, 9, 13, 10, 14, 11, 15 },  // Vertical block 1.


		{ 0, 4, 8, 12, 1, 5, 9, 13 },      // Horizontal block 0.
		{ 2, 6, 10, 14, 3, 7, 11, 15 }     // Horizontal block 1.
	};

	for (uint i = 1; i < 16; ++i) {
		if (src[i].bits != src[0].bits)
			return false;
	}

	__private float src_color_float[3] = { (float)(src->channels.b), (float)(src->channels.g), (float)(src->channels.r) };
	__private Color base = makeColor555(src_color_float);

	
	uchar var = dst[3] & ~0x02;

	dst[3] = var;
	dst[3] |= ((uchar)(1)) << 1;

	var = dst[3] & ~0x01;
	dst[3] &= var;
	dst[3] |= (uchar)(0);
	
	WriteColors555(dst, base, base);

	uchar best_tbl_idx = 0;
	uchar best_mod_idx = 0;

	uint best_mod_err = UINT32_MAX;

	for (uint tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		// Try all modifiers in the current table to find which one gives the
		// smallest error.
		for (uint mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = codeword_tables[tbl_idx][mod_idx];
			const Color color = makeColor(base, lum);

			uint mod_err = getColorError(*src, color);
			if (mod_err < best_mod_err) {
				best_tbl_idx = tbl_idx;
				best_mod_idx = mod_idx;
				best_mod_err = mod_err;

				if (mod_err == 0)
					break;  // We cannot do any better than this.
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
	for (uint i = 0; i < 2; ++i) {
		for (uint j = 0; j < 8; ++j) {
			// Obtain the texel number as specified in the standard.
			int texel_num = g_idx_to_num[i][j];
			pix_data |= msb << (texel_num + 16);
			pix_data |= lsb << (texel_num);
		}
	}

	WritePixelData(dst, pix_data);
	*error = 16 * best_mod_err;
	return true;
}


ulong compressBlock(uchar* dst, const Color* ver_src, Color* hor_src, ulong threshold) {

	__private ulong solid_error = 0;

	if (tryCompressSolidBlock(dst, ver_src, &solid_error)) {
		return solid_error;
	}
	
	__private Color* sub_block_src[4];
	sub_block_src[0] = ver_src; // = { ver_src, ver_src + 8, hor_src, hor_src + 8 };
	sub_block_src[1] = (uchar*)ver_src + 8 * 4;
	sub_block_src[2] = hor_src;
	sub_block_src[3] = (uchar*)hor_src + 8 * 4;

	__private Color sub_block_avg[4];
	__private uchar use_differential[2] = { 1, 1 };

	for (uint i = 0, j = 1; i < 4; i += 2, j += 2) {
		__private float avg_color_0[3];
		getAverageColor(sub_block_src[i], avg_color_0);
		Color avg_color_555_0 = makeColor555(avg_color_0);

		__private float avg_color_1[3];
		getAverageColor(sub_block_src[j], avg_color_1);
		Color avg_color_555_1 = makeColor555(avg_color_1);

		for (unsigned int light_idx = 0; light_idx < 3; ++light_idx) {
			int u = avg_color_555_0.components[light_idx] >> 3;
			int v = avg_color_555_1.components[light_idx] >> 3;

			int component_diff = v - u;
			if (component_diff  3) {
				use_differential[i / 2] = 0;
				sub_block_avg[i] = makeColor444(avg_color_0);
				sub_block_avg[j] = makeColor444(avg_color_1);
			}
			else {


				sub_block_avg[i] = avg_color_555_0;
				sub_block_avg[j] = avg_color_555_1;
			}
		}
	}

	uint sub_block_err[4] = { 0 };
	for (uint i = 0; i < 4; ++i) {
		for (uint j = 0; j < 8; ++j) {
			sub_block_err[i] += getColorError(sub_block_avg[i], sub_block_src[i][j]);
		}
	}

	bool flip =
		sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1];

	memset(dst, 0, 8);
	uchar var;
	var = ~0x02;
	if (use_differential[!!flip]) {
		dst[3] = var;
		dst[3] |= ((uchar)(1)) << 1;
	}
	else {
		dst[3] = var;
		dst[3] |= ((uchar)(0)) << 1;
	}

	var = ~0x01;
	if (flip) {
		dst[3] &= var;
		dst[3] |= (uchar)(1);
	}
	else {

		dst[3] &= var;
		dst[3] |= (uchar)(0);
	}
	
	uchar sub_block_off_0 = flip ? 2 : 0;
	uchar sub_block_off_1 = sub_block_off_0 + 1;

	if (use_differential[!!flip]) {
		WriteColors555(dst, sub_block_avg[sub_block_off_0],
			sub_block_avg[sub_block_off_1]);
	}
	else {
		WriteColors444(dst, sub_block_avg[sub_block_off_0],
			sub_block_avg[sub_block_off_1]);
	}

	ulong lumi_error1 = 0, lumi_error2 = 0;
	__const uchar g_idx_to_num[4][8] = {
		{ 0, 4, 1, 5, 2, 6, 3, 7 },        // Vertical block 0.
		{ 8, 12, 9, 13, 10, 14, 11, 15 },  // Vertical block 1.
		{ 0, 4, 8, 12, 1, 5, 9, 13 },      // Horizontal block 0.
		{ 2, 6, 10, 14, 3, 7, 11, 15 }     // Horizontal block 1.
	};

	
	
	// Compute luminance for the first sub block.
	lumi_error1 = computeLuminance(dst, sub_block_src[sub_block_off_0],
		sub_block_avg[sub_block_off_0], 0,
		g_idx_to_num[sub_block_off_0],
		threshold);
	
	// Compute luminance for the second sub block.
	lumi_error2 = computeLuminance(dst, sub_block_src[sub_block_off_1],
		sub_block_avg[sub_block_off_1], 1,
		g_idx_to_num[sub_block_off_1],
		threshold);

	return lumi_error1 + lumi_error2;
}

__kernel void
compress(__global uchar* src,
        __global uchar* dst,
        int width,
        int height,
	__global uchar* compressed_error)
{
	const int row = get_global_id(0); // Local row ID (max: TS)
	const int col = get_global_id(1); // Local col ID (max: TS)
	
	int inc = 0;

	uchar local_src[64];

	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++) {
			local_src[inc]		= src[4 * (row * width / 4 + col) + i * width + j * 4];
			local_src[inc + 1]	= src[4 * (row * width / 4 + col) + i * width + j * 4 + 1];
			local_src[inc + 2]	= src[4 * (row * width / 4 + col) + i * width + j * 4 + 2];
			local_src[inc + 3]	= src[4 * (row * width / 4 + col) + i * width + j * 4 + 3];			
			inc+=4;
		}

	//START
			
	__private Color ver_blocks[16];
	__private Color hor_blocks[16];
	__private uchar local_dst[8];
	 
	Color row0[4];
	Color row1[4];
	Color row2[4];
	Color row3[4];
	
	memcpy((uchar*)row0, local_src + 0, 16);
	memcpy((uchar*)row1, local_src + 16, 16);
	memcpy((uchar*)row2, local_src + 32, 16);
	memcpy((uchar*)row3, local_src + 48, 16);

	memcpy((uchar*)ver_blocks, row0, 8);
	memcpy((uchar*)ver_blocks + 2 * 4, row1, 8);
	memcpy((uchar*)ver_blocks + 4 * 4, row2, 8);
	memcpy((uchar*)ver_blocks + 6 * 4, row3, 8);
	memcpy((uchar*)ver_blocks + 8 * 4, row0 + 2 * 4, 8);
	memcpy((uchar*)ver_blocks + 10 * 4, row1 + 2 * 4, 8);
	memcpy((uchar*)ver_blocks + 12 * 4, row2 + 2 * 4, 8);
	memcpy((uchar*)ver_blocks + 14 * 4, row3 + 2 * 4, 8);

	memcpy((uchar*)hor_blocks, row0, 16);
	memcpy((uchar*)hor_blocks + 4 * 4, row1, 16);
	memcpy((uchar*)hor_blocks + 8 * 4, row2, 16);
	memcpy((uchar*)hor_blocks + 12 * 4, row3, 16)
	
	//FINISH
	ulong old_error;
	old_error = compressBlock(local_dst, ver_blocks, hor_blocks, INT32_MAX);
	
	for (int i = 0; i < 8; i++)
		dst[row * width / 4 * 8 + col * 8 + i] = local_dst[i];
}

>>>> file: texture_compress_skl.cpp
#include "compress.hpp"

#include 
#include 
#include 
#include 
#include 
#include "helper.hpp"

#define uint unsigned int



using namespace std;

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

	/* get num of available OpenCL platforms */
	CL_ERR( clGetPlatformIDs(0, NULL, &platform_num));
	platform_list = new cl_platform_id[platform_num];
	DIE(platform_list == NULL, "alloc platform_list");

	/* get all available OpenCL platforms */
	CL_ERR( clGetPlatformIDs(platform_num, platform_list, NULL));
	cout << "Platforms found: " << platform_num << endl;

	/* list all platforms and VENDOR/VERSION properties */
	for(uint platf=0; platf<platform_num; platf++)
	{
		/* get attribute CL_PLATFORM_VENDOR */
		CL_ERR( clGetPlatformInfo(platform_list[platf],
				CL_PLATFORM_VENDOR, 0, NULL, &attr_size));
		attr_data = new cl_char[attr_size];
		DIE(attr_data == NULL, "alloc attr_data");

		/* get data CL_PLATFORM_VENDOR */
		CL_ERR( clGetPlatformInfo(platform_list[platf],
				CL_PLATFORM_VENDOR, attr_size, attr_data, NULL));
		cout << "Platform " << platf << " " << attr_data << " ";
		delete[] attr_data;

		/* get attribute size CL_PLATFORM_VERSION */
		CL_ERR( clGetPlatformInfo(platform_list[platf],
				CL_PLATFORM_VERSION, 0, NULL, &attr_size));
		attr_data = new cl_char[attr_size];
		DIE(attr_data == NULL, "alloc attr_data");

		/* get data size CL_PLATFORM_VERSION */
		CL_ERR( clGetPlatformInfo(platform_list[platf],
				CL_PLATFORM_VERSION, attr_size, attr_data, NULL));
		cout << attr_data << endl;
		delete[] attr_data;

		/* no valid platform found */
		platform = platform_list[platf];
		DIE(platform == 0, "platform selection");

		/* get num of available OpenCL devices type ALL on the selected platform */
		if(clGetDeviceIDs(platform, 
			CL_DEVICE_TYPE_GPU, 0, NULL, &device_num) == CL_DEVICE_NOT_FOUND) {
			device_num = 0;
			continue;
		}

		device_list = new cl_device_id[device_num];
		DIE(device_list == NULL, "alloc devices");

		/* get all available OpenCL devices type ALL on the selected platform */
		CL_ERR( clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU,
			  device_num, device_list, NULL));
		cout << "\tDevices found " << device_num  << endl;

		/* list all devices and TYPE/VERSION properties */
		for(uint dev=0; dev<device_num; dev++)
		{
			/* get attribute size */
			CL_ERR( clGetDeviceInfo(device_list[dev], CL_DEVICE_NAME,
				0, NULL, &attr_size));
			attr_data = new cl_char[attr_size];
			DIE(attr_data == NULL, "alloc attr_data");

			/* get attribute CL_DEVICE_NAME */
			CL_ERR( clGetDeviceInfo(device_list[dev], CL_DEVICE_NAME,
				attr_size, attr_data, NULL));
			cout << "\tDevice " << dev << " " << attr_data << " ";
			delete[] attr_data;

			/* get attribute size */
			CL_ERR( clGetDeviceInfo(device_list[dev], CL_DEVICE_VERSION,
				0, NULL, &attr_size));
			attr_data = new cl_char[attr_size];
			DIE(attr_data == NULL, "alloc attr_data");

			/* get attribute CL_DEVICE_VERSION */
			CL_ERR( clGetDeviceInfo(device_list[dev], CL_DEVICE_VERSION,
				attr_size, attr_data, NULL));
			cout << attr_data; 
			delete[] attr_data;

			if (device_list[dev] != NULL) {			
				/* select device based on cli arguments */
				device = device_list[dev];
				cout << " <--- SELECTED " << endl;		
				break;
			}	
		}
	}

	delete[] platform_list;
	delete[] device_list;
}

TextureCompressor::TextureCompressor() {
	cl_device_id device2;
	uint platform_select = 0;
	uint device_select = 0;
	cl_int ret;
	string kernel_src;

	/* search and select platform/devices in OpenCL */
	gpu_find(device2, platform_select, device_select);
	this->device = device2;

	context = clCreateContext(0, 1, &device, NULL, NULL, &ret);	
	CL_ERR( ret );

	command_queue = clCreateCommandQueue(context, device, 0, &ret);
	CL_ERR( ret );

	read_kernel("skl_device.cl", kernel_src);
	const char* kernel_c_str = kernel_src.c_str();



	program = clCreateProgramWithSource(context, 1, &kernel_c_str, NULL, &ret);
	CL_ERR( ret );

	ret = clBuildProgram(program, 1, &device, "", NULL, NULL);
	CL_COMPILE_ERR( ret, program, device );

	kernel = clCreateKernel(program, "compress", &ret);
	CL_ERR( ret );

}

 // constructor/Users/grigore.lupescu/Desktop/RESEARCH/asc/teme/tema3/2018/Tema3-schelet/src/compress.cpp
TextureCompressor::~TextureCompressor() { }	// destructor
	
unsigned long TextureCompressor::compress(const uint8_t* src, uint8_t* dst, int width, int height)
{
	// TODO
	cl_mem sImg, cImg;
	size_t global[2];
        size_t local[2];
	int ret = 0;

        cout << "Begin sending to GPU" << endl;

	sImg = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(uint8_t) * width * height * 4, NULL, NULL);

	cImg = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(uint8_t) * width * height / 2, NULL, NULL); 
	
	/* Write the buffer to GPU*/
	cout << "BEGIN WRITING IMAGE TO GPU" << endl << endl;
	ret = clEnqueueWriteBuffer(command_queue, sImg, CL_TRUE, 0, sizeof(uint8_t) * width * height * 4, src, 0, NULL, NULL);

	cout << "END WRITING IMAGE TO GPU" << endl << endl;

	int err = 0;
	cl_mem err_buff;
	err_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_ulong) * width * height / 16, NULL, NULL);

	/* Set openCL arguments*/
	CL_ERR(clSetKernelArg(kernel, 0, sizeof(cl_mem), &sImg));
	CL_ERR(clSetKernelArg(kernel, 1, sizeof(cl_mem), &cImg));
	
	CL_ERR(clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&width));
	CL_ERR(clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&height));
	CL_ERR(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&err_buff));
	
	
	/* Set global and local work size */
	global[0] = (size_t) width / 4;
	global[1] = (size_t) height / 4;
        local[0] = 1;
        local[1] = 1;

	/* Enqueue the kernel. */
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global, local, 0, NULL, NULL);
	
	/* Wait for the command queue */
	ret = clFinish(command_queue);


	cout << "BEGIN READ FROM GPU" << endl << endl;
	/* Read into the buffer from GPU*/
	ret = clEnqueueReadBuffer(command_queue, cImg, CL_TRUE, 0, sizeof(uint8_t) * width  * height / 2 , dst, 0, NULL, NULL );
	ret += clEnqueueReadBuffer(command_queue, err_buff, CL_TRUE, 0, sizeof(int), &err, 0, NULL, NULL);

	cout << "END READ FROM GPU" << endl << endl;

	clReleaseProgram(program);
        clReleaseKernel(kernel);
        clReleaseMemObject(sImg);
        clReleaseMemObject(cImg);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context);
	

	return err;
}
