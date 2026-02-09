/**
 * @file helper.cpp
 * @brief Semantic documentation for helper.cpp. 
 *        This is a placeholder. Detailed semantic analysis will be applied later.
 */

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
>>>> file: kernel_andreea.cl
#define ALIGNAS(X)	__attribute__((aligned(X)))
#define UINT32_MAX  0xffffffff
#define INT32_MAX   0x7fffffff

union Color {
	struct BgraColorType {
		unsigned char b;
		unsigned char g;
		unsigned char r;
		unsigned char a;
	} channels;
	unsigned char components[4];


	unsigned int bits;
};

inline unsigned char clamp_char(unsigned char val, unsigned char min,
						   unsigned char max) {
	return val  max ? max : val);
}

inline int clamp_int(int val, int min, int max) {
	return val  max ? max : val);
}

inline unsigned char round_to_5_bits(float val) {
	return clamp_char(val * 31.0f / 255.0f + 0.5f, 0, 31);
}

inline unsigned char round_to_4_bits(float val) {
	return clamp_char(val * 15.0f / 255.0f + 0.5f, 0, 15);
}

// Codeword tables.
// See: Table 3.17.2
ALIGNAS(16) __constant short g_codeword_tables[8][4] = {
		{-8, -2, 2, 8},
		{-17, -5, 5, 17},
		{-29, -9, 9, 29},
		{-42, -13, 13, 42},
		{-60, -18, 18, 60},


		{-80, -24, 24, 80},
		{-106, -33, 33, 106},
		{-183, -47, 47, 183}};

// Maps modifier indices to pixel index values.
// See: Table 3.17.3
__constant unsigned char g_mod_to_pix[4] = {3, 2, 0, 1};

// The ETC1 specification index texels as follows:
// [a][e][i][m]     [ 0][ 4][ 8][12]
// [b][f][j][n]  [ 1][ 5][ 9][13]
// [c][g][k][o]     [ 2][ 6][10][14]
// [d][h][l][p]     [ 3][ 7][11][15]

// [ 0][ 1][ 2][ 3]     [ 0][ 1][ 4][ 5]
// [ 4][ 5][ 6][ 7]  [ 8][ 9][12][13]
// [ 8][ 9][10][11]     [ 2][ 3][ 6][ 7]
// [12][13][14][15]     [10][11][14][15]

// However, when extracting sub blocks from BGRA data the natural array
// indexing order ends up different:
// vertical0: [a][e][b][f]  horizontal0: [a][e][i][m]
//            [c][g][d][h]               [b][f][j][n]
// vertical1: [i][m][j][n]  horizontal1: [c][g][k][o]
//            [k][o][l][p]               [d][h][l][p]

// In order to translate from the natural array indices in a sub block to the
// indices (number) used by specification and hardware we use this table.
__constant unsigned char g_idx_to_num[4][8] = {
		{0, 4, 1, 5, 2, 6, 3, 7},        // Vertical block 0.
		{8, 12, 9, 13, 10, 14, 11, 15},  // Vertical block 1.
		{0, 4, 8, 12, 1, 5, 9, 13},      // Horizontal block 0.
		{2, 6, 10, 14, 3, 7, 11, 15}     // Horizontal block 1.
};

// Constructs a color from a given base color and luminance value.
inline union Color makeColor(const union Color base, short lum) {
	int b = (int)(base.channels.b) + lum;
	int g = (int)(base.channels.g) + lum;
	int r = (int)(base.channels.r) + lum;


	union Color color;
	color.channels.b = (unsigned char)(clamp_int(b, 0, 255));
	color.channels.g = (unsigned char)(clamp_int(g, 0, 255));
	color.channels.r = (unsigned char)(clamp_int(r, 0, 255));
	return color;
}

// Calculates the error metric for two colors. A small error signals that the
// colors are similar to each other, a large error the signals the opposite.
inline int getColorError(const union Color u, const union Color v) {
#ifdef USE_PERCEIVED_ERROR_METRIC
	float delta_b = (float)(u.channels.b) - v.channels.b;
	float delta_g = (float)(u.channels.g) - v.channels.g;
	float delta_r = (float)(u.channels.r) - v.channels.r;
	return (int)(0.299f * delta_b * delta_b +
								 0.587f * delta_g * delta_g +
								 0.114f * delta_r * delta_r);
#else
	int delta_b = (int)(u.channels.b) - v.channels.b;
	int delta_g = (int)(u.channels.g) - v.channels.g;
	int delta_r =(int)(u.channels.r) - v.channels.r;
	return delta_b * delta_b + delta_g * delta_g + delta_r * delta_r;
#endif
}

void myMemcpy(unsigned char *dest, unsigned char *src, size_t n)
{
	/* copy contents of src to dest */
	for (int i = 0; i < n; i++)
		dest[i] = src[i];
}

inline void WriteColors444(__global unsigned char* block,
						   const union Color color0,
						   const union Color color1) {
	// Write output color for BGRA textures.
	block[0] = (color0.channels.r & 0xf0) | (color1.channels.r >> 4);
	block[1] = (color0.channels.g & 0xf0) | (color1.channels.g >> 4);
	block[2] = (color0.channels.b & 0xf0) | (color1.channels.b >> 4);
}

inline void WriteColors555(__global unsigned char* block,
						   const union Color color0,


						   const union Color color1) {
	// Table for conversion to 3-bit two complement format.
	const unsigned char two_compl_trans_table[8] = {
			4,  // -4 (100b)
			5,  // -3 (101b)
			6,  // -2 (110b)
			7,  // -1 (111b)
			0,  //  0 (000b)
			1,  //  1 (001b)
			2,  //  2 (010b)
			3,  //  3 (011b)
	};

	short delta_r =
			(short)(color1.channels.r >> 3) - (color0.channels.r >> 3);
	short delta_g =
			(short)(color1.channels.g >> 3) - (color0.channels.g >> 3);
	short delta_b =
			(short)(color1.channels.b >> 3) - (color0.channels.b >> 3);

	// Write output color for BGRA textures.
	block[0] = (color0.channels.r & 0xf8) | two_compl_trans_table[delta_r + 4];
	block[1] = (color0.channels.g & 0xf8) | two_compl_trans_table[delta_g + 4];


	block[2] = (color0.channels.b & 0xf8) | two_compl_trans_table[delta_b + 4];
}

inline void WriteCodewordTable(__global unsigned char* block,
							   unsigned char sub_block_id,
							   unsigned char table) {

	unsigned char shift = (2 + (3 - sub_block_id * 3));
	block[3] &= ~(0x07 << shift);
	block[3] |= table << shift;
}

inline void WritePixelData(__global unsigned char* block, unsigned int
pixel_data) {
	block[4] |= pixel_data >> 24;
	block[5] |= (pixel_data >> 16) & 0xff;
	block[6] |= (pixel_data >> 8) & 0xff;
	block[7] |= pixel_data & 0xff;
}

inline void WriteFlip(__global unsigned char* block, bool flip) {
	block[3] &= ~0x01;
	block[3] |= (unsigned char)(flip);
}

inline void WriteDiff(__global unsigned char* block, bool diff) {
	block[3] &= ~0x02;
	block[3] |= (unsigned char)(diff) << 1;
}



inline void ExtractBlock(unsigned char* dst, const unsigned char* src, int width) {
	for (int j = 0; j < 4; ++j) {
		myMemcpy(&dst[j * 4 * 4], src, 4 * 4);
		src += width * 4;
	}
}

// Compress and rounds BGR888 into BGR444. The resulting BGR444 color is
// expanded to BGR888 as it would be in hardware after decompression. The
// actual 444-bit data is available in the four most significant bits of each
// channel.
inline union Color makeColor444(const float* bgr) {
	unsigned char b4 = round_to_4_bits(bgr[0]);
	unsigned char g4 = round_to_4_bits(bgr[1]);
	unsigned char r4 = round_to_4_bits(bgr[2]);
	union Color bgr444;
	bgr444.channels.b = (b4 << 4) | b4;
	bgr444.channels.g = (g4 << 4) | g4;
	bgr444.channels.r = (r4 << 4) | r4;
	// Added to distinguish between expanded 555 and 444 colors.
	bgr444.channels.a = 0x44;
	return bgr444;
}

// Compress and rounds BGR888 into BGR555. The resulting BGR555 color is
// expanded to BGR888 as it would be in hardware after decompression. The
// actual 555-bit data is available in the five most significant bits of each
// channel.
inline union Color makeColor555(const float* bgr) {
	unsigned char b5 = round_to_5_bits(bgr[0]);
	unsigned char g5 = round_to_5_bits(bgr[1]);
	unsigned char r5 = round_to_5_bits(bgr[2]);
	union Color bgr555;
	bgr555.channels.b = (b5 > 2);
	bgr555.channels.g = (g5 > 2);
	bgr555.channels.r = (r5 > 2);
	// Added to distinguish between expanded 555 and 444 colors.
	bgr555.channels.a = 0x55;
	return bgr555;
}



void getAverageColor(const union Color* src, float* avg_color)
{
	unsigned int sum_b = 0, sum_g = 0, sum_r = 0;

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
							   const union Color* src,


							   const union Color base,
							   int sub_block_id,
							   __constant unsigned char* idx_to_num_tab,
							   unsigned long threshold)
{
	unsigned int best_tbl_err = threshold;
	unsigned char best_tbl_idx = 0;
	unsigned char best_mod_idx[8][8];  // [table][texel]

	// Try all codeword tables to find the one giving the best results for this
	// block.
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		// Pre-compute all the candidate colors; combinations of the base color and
		// all available luminance values.
		union Color candidate_color[4];  // [modifier]


		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			candidate_color[mod_idx] = makeColor(base, lum);
		}

		unsigned int tbl_err = 0;

		for (unsigned int i = 0; i < 8; ++i) {
			// Try all modifiers in the current table to find which one gives the
			// smallest error.
			unsigned int best_mod_err = threshold;
			for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {


				const union Color color = candidate_color[mod_idx];

				unsigned int mod_err = getColorError(src[i], color);
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

	unsigned int pix_data = 0;

	for (unsigned int i = 0; i < 8; ++i) {
		unsigned char mod_idx = best_mod_idx[best_tbl_idx][i];
		unsigned char pix_idx = g_mod_to_pix[mod_idx];

		unsigned int lsb = pix_idx & 0x1;
		unsigned int msb = pix_idx >> 1;

		// Obtain the texel number as specified in the standard.
		int texel_num = idx_to_num_tab[i];
		pix_data |= msb << (texel_num + 16);
		pix_data |= lsb << (texel_num);
	}

	WritePixelData(block, pix_data);

	return best_tbl_err;
}

void myMemset(__global unsigned char *b, int c, int len)
{
	int i = 0;

	while(len > 0)
	{
		*b = c;
		b++;
		len--;
	}
}

/**
 * Tries to compress the block under the assumption that it's a single color
 * block. If it's not the function will bail out without writing anything to
 * the destination buffer.
 */
bool tryCompressSolidBlock(__global unsigned char* dst,
						   const union Color* src,
						   unsigned long* error)
{
	for (unsigned int i = 1; i < 16; ++i) {
		if (src[i].bits != src[0].bits)
			return false;
	}

	// Clear destination buffer so that we can "or" in the results.
	myMemset(dst, 0, 8);

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

	// Try all codeword tables to find the one giving the best results for this
	// block.
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		// Try all modifiers in the current table to find which one gives the
		// smallest error.
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];


			const union Color color = makeColor(base, lum);

			unsigned int mod_err = getColorError(*src, color);
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

	unsigned char pix_idx = g_mod_to_pix[best_mod_idx];
	unsigned int lsb = pix_idx & 0x1;
	unsigned int msb = pix_idx >> 1;

	unsigned int pix_data = 0;
	for (unsigned int i = 0; i < 2; ++i) {
		for (unsigned int j = 0; j < 8; ++j) {
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
unsigned long compressBlock(__global unsigned char* dst, const union
		Color* ver_src, const union Color* hor_src, unsigned long threshold)
{
	unsigned long solid_error = 0;
	if (tryCompressSolidBlock(dst, ver_src, &solid_error)) {
		return solid_error;
	}

	const union Color* sub_block_src[4] = {ver_src, ver_src + 8, hor_src,
								   hor_src + 8};

	union Color sub_block_avg[4];
	bool use_differential[2] = {true, true};

	// Compute the average color for each sub block and determine if differential
	// coding can be used.
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

	// Compute the error of each sub block before adjusting for luminance. These
	// error values are later used for determining if we should flip the sub
	// block or not.
	unsigned int sub_block_err[4] = {0};
	for (unsigned int i = 0; i < 4; ++i) {
		for (unsigned int j = 0; j < 8; ++j) {
			sub_block_err[i] += getColorError(sub_block_avg[i], sub_block_src[i][j]);
		}
	}

	bool flip =
			sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1];

	// Clear destination buffer so that we can "or" in the results.
	myMemset(dst, 0, 8);



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

__kernel void gpu_kernel(const int width,
                         const int height,
                         __global unsigned char *in_buf,
                         __global unsigned char *out_buf){

	union Color ver_blocks[16];
	union Color hor_blocks[16];

	int column_index = get_global_id(0);
	int row_index = get_global_id(1);

	in_buf += 16 * row_index * width + 16 * column_index;
	out_buf += 2 * row_index * width + 8 * column_index;

	__global union Color* row0 = (__global union Color*) in_buf;
	__global union Color* row1 = row0 + width;
	__global union Color* row2 = row1 + width;
	__global union Color* row3 = row2 + width;

	myMemcpy(ver_blocks, row0, 8);
	myMemcpy(ver_blocks + 2, row1, 8);
	myMemcpy(ver_blocks + 4, row2, 8);
	myMemcpy(ver_blocks + 6, row3, 8);
	myMemcpy(ver_blocks + 8, row0 + 2, 8);
	myMemcpy(ver_blocks + 10, row1 + 2, 8);
	myMemcpy(ver_blocks + 12, row2 + 2, 8);
	myMemcpy(ver_blocks + 14, row3 + 2, 8);

	myMemcpy(hor_blocks, row0, 16);
	myMemcpy(hor_blocks + 4, row1, 16);
	myMemcpy(hor_blocks + 8, row2, 16);
	myMemcpy(hor_blocks + 12, row3, 16);

	compressBlock(out_buf, ver_blocks, hor_blocks, INT32_MAX);
}>>>> file: texture_compress_andreea.cpp
#include "compress.hpp"
#include 
#include 
#include 
#include 
#include 
#include "helper.hpp"

using namespace std;

/* Retrieve GPU device */
void gpu_find(cl_device_id &device, cl_platform_id *platform_ids,
			  cl_device_id *device_ids)
{
	cl_platform_id platform;
	cl_uint platform_num = 0, device_num = 0;
	size_t attr_size = 0;
	cl_char *attr_data = NULL;

	/* get num of available OpenCL platforms */
	CL_ERR(clGetPlatformIDs(0, NULL, &platform_num));
	platform_ids = new cl_platform_id[platform_num];
	DIE(platform_ids == NULL, "alloc platform_ids");

	/* get all available OpenCL platforms */
	CL_ERR(clGetPlatformIDs(platform_num, platform_ids, NULL));
	cout << "Platforms found: " << platform_num << endl;

	/* list all platforms and VENDOR/VERSION properties */
	for (uint platf = 0; platf < platform_num; platf++) {
		/* get attribute CL_PLATFORM_VENDOR */
		CL_ERR(clGetPlatformInfo(platform_ids[platf],
								 CL_PLATFORM_VENDOR, 0, NULL, &attr_size));
		attr_data = new cl_char[attr_size];
		DIE(attr_data == NULL, "alloc attr_data");

		/* get data CL_PLATFORM_VENDOR */
		CL_ERR(clGetPlatformInfo(platform_ids[platf],
								 CL_PLATFORM_VENDOR, attr_size, attr_data,
								 NULL));
		cout << "Platform " << platf << " " << attr_data << " ";
		delete[] attr_data;

		/* get attribute size CL_PLATFORM_VERSION */
		CL_ERR(clGetPlatformInfo(platform_ids[platf],
								 CL_PLATFORM_VERSION, 0, NULL,
								 &attr_size));
		attr_data = new cl_char[attr_size];
		DIE(attr_data == NULL, "alloc attr_data");

		/* get data size CL_PLATFORM_VERSION */
		CL_ERR(clGetPlatformInfo(platform_ids[platf],
								 CL_PLATFORM_VERSION, attr_size, attr_data,
								 NULL));
		cout << attr_data << endl;
		delete[] attr_data;

		/* no valid platform found */
		platform = platform_ids[platf];
		DIE(platform == 0, "platform selection");

		/* get num of available OpenCL devices type GPU on the selected platform */
		CL_ERR(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL,
							  &device_num));

		/* if I have at least one OpenCL devices type GPU on this
		 * platform */
		if (device_num > 0) {
			device_ids = new cl_device_id[device_num];
			DIE(device_ids == NULL, "alloc devices");

			/* get all available OpenCL devices type GPU on the selected platform */
			CL_ERR(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU,
								  device_num, device_ids, NULL));
			cout << "\tDevices found " << device_num << endl;

			/* list the first device found and TYPE/VERSION properties */
			device = device_ids[0];

			/* get attribute size */
			CL_ERR(clGetDeviceInfo(device_ids[0], CL_DEVICE_NAME,
								   0, NULL, &attr_size));
			attr_data = new cl_char[attr_size];
			DIE(attr_data == NULL, "alloc attr_data");

			/* get attribute CL_DEVICE_NAME */
			CL_ERR(clGetDeviceInfo(device_ids[0], CL_DEVICE_NAME,
								   attr_size, attr_data, NULL));
			cout << "\tFirst device " << " " << attr_data << " ";
			delete[] attr_data;

			/* get attribute size */
			CL_ERR(clGetDeviceInfo(device_ids[0], CL_DEVICE_VERSION,
								   0, NULL, &attr_size));
			attr_data = new cl_char[attr_size];
			DIE(attr_data == NULL, "alloc attr_data");

			/* get attribute CL_DEVICE_VERSION */
			CL_ERR(clGetDeviceInfo(device_ids[0], CL_DEVICE_VERSION,
								   attr_size, attr_data, NULL));
			cout << attr_data;
			delete[] attr_data;

			cout << endl;
			break;
		}
	}

	delete[] platform_ids;
	delete[] device_ids;
}

void gpu_profile_kernel(cl_device_id device_id, int width, int height,
						const uint8_t *src, uint8_t *dst, cl_context
						context, cl_program program, cl_kernel kernel,
						cl_command_queue commands)
{
	size_t global[2];
	size_t local[2];
	string kernel_src;
	cl_uint nd;
	cl_mem in;
	cl_mem out;
	int i, ret;
	int Ndim;

	/* create a context for the device */
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &ret);
	CL_ERR(ret);

	/* create a command-queue for the device */
	commands = clCreateCommandQueue(context, device_id,
									CL_QUEUE_PROFILING_ENABLE, &ret);
	CL_ERR(ret);

	/* width, height correspond to pixels, and a pixel has red green
	 * blue and alpha chanel  */
	Ndim = width * height * 4;

	/* set up the buffers */
	in = clCreateBuffer(context, CL_MEM_READ_ONLY,
						sizeof(uint8_t) * Ndim, NULL, NULL);

	out = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
						 sizeof(uint8_t) * Ndim / 8, NULL, NULL);

	/* retrieve kernel source */
	read_kernel("kernel_andreea.cl", kernel_src);
	const char *kernel_c_str = kernel_src.c_str();

	/* create the compute program from the source buffer */
	program = clCreateProgramWithSource(context, 1,
										(const char **) &kernel_c_str,
										NULL, &ret);
	CL_ERR(ret);

	/* build the program */
	ret = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	CL_COMPILE_ERR(ret, program, device_id);

	/* create the compute kernel from the program */
	kernel = clCreateKernel(program, "gpu_kernel", &ret);
	CL_ERR(ret);

	/* set the arguments to our compute kernel */
	ret = 0;
	ret = clSetKernelArg(kernel, 0, sizeof(int), &width);
	ret |= clSetKernelArg(kernel, 1, sizeof(int), &height);
	ret |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &in);
	ret |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &out);

	/* write the src into compute device memory */
	ret = clEnqueueWriteBuffer(commands, in, CL_TRUE, 0,
							   sizeof(uint8_t) * Ndim, src, 0, NULL, NULL);

	/* execute the kernel over the entire range of image components */
	global[0] = (size_t) width / 4; // number of workers, block size = 4
	global[1] = (size_t) height / 4;

	cl_event prof_event;
	/* enqueues the commands to execute kernel on device*/
	ret = clEnqueueNDRangeKernel(commands, kernel, 2, NULL,
								 global, NULL, 0, NULL, &prof_event);

	/* wait for the commands to complete before reading back results */
	clFinish(commands);
	cl_ulong run_time = (cl_ulong) 0;
	cl_ulong ev_start_time = (cl_ulong) 0;
	cl_ulong ev_end_time = (cl_ulong) 0;
	size_t ret_size;

	/* read back the results from the compute device */


	ret = clEnqueueReadBuffer(commands, out, CL_TRUE, 0,
							  sizeof(uint8_t) * Ndim / 8, dst, 0, NULL,
							  NULL);

	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseMemObject(in);
	clReleaseMemObject(out);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);
}

/* Constructor */
TextureCompressor::TextureCompressor()
{
	/* search and select platform and first device GPU in OpenCL */
	gpu_find(device, platform_ids, device_ids);
}

/* Destructor */
TextureCompressor::~TextureCompressor()
{}

/* Compress function */
unsigned long TextureCompressor::compress(const uint8_t *src,
										  uint8_t *dst,
										  int width,
										  int height)
{

	gpu_profile_kernel(device, width, height, src, dst, context,
					   program, kernel, command_queue);

	return 0;
}
