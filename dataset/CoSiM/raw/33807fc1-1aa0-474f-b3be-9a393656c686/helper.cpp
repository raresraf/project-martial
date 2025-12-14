
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
>>>> file: kernel_maria.cl
typedef union Color {
	struct BgraColorType {
		unsigned char b;
		unsigned char g;
		unsigned char r;
		unsigned char a;
	} channels;
	unsigned char components[4];
	unsigned int bits;


} Color;


inline unsigned char clamp_renamed(unsigned char val, unsigned char min, unsigned char max) {
	return val  max ? max : val);
}


inline int clamp_renamed_int(int val, int min, int max) {
	return val  max ? max : val);
}

inline unsigned char round_to_5_bits(float val) {
	return clamp_renamed((unsigned char)(val * 31.0f / 255.0f + 0.5f), (unsigned char)0, (unsigned char)31);
}

inline unsigned char round_to_4_bits(float val) {
	return clamp_renamed((unsigned char)(val * 15.0f / 255.0f + 0.5f), (unsigned char)0, (unsigned char)15);
}

// Codeword tables.
// See: Table 3.17.2
__constant short g_codeword_tables[8][4] = {
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
inline Color makeColor(const Color base, short lum) {
	int b = (int)(base.channels.b) + lum;
	int g = (int)(base.channels.g) + lum;
	int r = (int)(base.channels.r) + lum;


	Color color;
	color.channels.b = (unsigned char)(clamp_renamed_int(b, 0, 255));
	color.channels.g = (unsigned char)(clamp_renamed_int(g, 0, 255));
	color.channels.r = (unsigned char)(clamp_renamed_int(r, 0, 255));
	return color;
}

// Calculates the error metric for two colors. A small error signals that the
// colors are similar to each other, a large error the signals the opposite.
inline unsigned int getColorError(const Color u, const Color v) {
#ifdef USE_PERCEIVED_ERROR_METRIC
	float delta_b = (float)(u.channels.b) - v.channels.b;
	float delta_g = (float)(u.channels.g) - v.channels.g;
	float delta_r = (float)(u.channels.r) - v.channels.r;
	return (unsigned int)(0.299f * delta_b * delta_b +
								 0.587f * delta_g * delta_g +
								 0.114f * delta_r * delta_r);
#else
	int delta_b = (int)(u.channels.b) - v.channels.b;
	int delta_g = (int)(u.channels.g) - v.channels.g;
	int delta_r = (int)(u.channels.r) - v.channels.r;
	return delta_b * delta_b + delta_g * delta_g + delta_r * delta_r;
#endif
}

inline void WriteColors444(global unsigned char* block,
						   const Color color0,
						   const Color color1) {
	// Write output color for BGRA textures.
	block[0] = (color0.channels.r & 0xf0) | (color1.channels.r >> 4);
	block[1] = (color0.channels.g & 0xf0) | (color1.channels.g >> 4);
	block[2] = (color0.channels.b & 0xf0) | (color1.channels.b >> 4);
}

static inline void WriteColors555(global unsigned char* block,
						   const Color color0,


						   const Color color1) {
	// Table for conversion to 3-bit two complement format.
	unsigned char two_compl_trans_table[8] = {
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

inline void WriteCodewordTable(global unsigned char* block,
							   unsigned char sub_block_id,
							   unsigned char table) {
	
	unsigned char shift = (2 + (3 - sub_block_id * 3));
	block[3] &= ~(0x07 << shift);
	block[3] |= table << shift;
}

inline void WritePixelData(global unsigned char* block, unsigned int pixel_data) {
	block[4] |= pixel_data >> 24;
	block[5] |= (pixel_data >> 16) & 0xff;
	block[6] |= (pixel_data >> 8) & 0xff;
	block[7] |= pixel_data & 0xff;
}

inline void WriteFlip(global unsigned char* block, bool flip) {
	block[3] &= ~0x01;
	block[3] |= (unsigned char)(flip);
}

inline void WriteDiff(global unsigned char* block, bool diff) {
	block[3] &= ~0x02;
	block[3] |= (unsigned char)(diff) << 1;
}

void my_memcpy(void *dst, const void *src, size_t size)
{
   char *pdst = dst;
   const char *psrc = src;
   for (size_t i=0; i<size; i++) 
      pdst[i] = psrc[i];
}



inline void ExtractBlock(unsigned char* dst, const unsigned char* src, int width) {
	for (int j = 0; j < 4; ++j) {
		my_memcpy(&dst[j * 4 * 4], src, 4 * 4);
		src += width * 4;
	}
}

// Compress and rounds BGR888 into BGR444. The resulting BGR444 color is
// expanded to BGR888 as it would be in hardware after decompression. The
// actual 444-bit data is available in the four most significant bits of each
// channel.
inline Color makeColor444(const float* bgr) {
	unsigned char b4 = round_to_4_bits(bgr[0]);
	unsigned char g4 = round_to_4_bits(bgr[1]);
	unsigned char r4 = round_to_4_bits(bgr[2]);
	Color bgr444;
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
inline Color makeColor555(const float* bgr) {
	unsigned char b5 = round_to_5_bits(bgr[0]);
	unsigned char g5 = round_to_5_bits(bgr[1]);
	unsigned char r5 = round_to_5_bits(bgr[2]);
	Color bgr555;
	bgr555.channels.b = (b5 > 2);
	bgr555.channels.g = (g5 > 2);
	bgr555.channels.r = (r5 > 2);
	// Added to distinguish between expanded 555 and 444 colors.
	bgr555.channels.a = 0x55;
	return bgr555;
}
	


void getAverageColor(const Color* src, float* avg_color)
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
	
unsigned long computeLuminance(global unsigned char* block,
						   const Color* src,


						   const Color base,
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
		Color candidate_color[4];  // [modifier]


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


				const Color color = candidate_color[mod_idx];
				
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


/**
 * Tries to compress the block under the assumption that it's a single color
 * block. If it's not the function will bail out without writing anything to
 * the destination buffer.
 */
bool tryCompressSolidBlock(global unsigned char* dst,
						   const Color* src,
						   unsigned long* error)
{
	for (unsigned int i = 1; i < 16; ++i) {
		if (src[i].bits != src[0].bits)
			return false;
	}
	
	// Clear destination buffer so that we can "or" in the results.

   	for (size_t i=0; i<8; i++) 
    	dst[i] = (unsigned char)0;
	
	float src_color_float[3] = {(float)(src->channels.b),
		(float)(src->channels.g),
		(float)(src->channels.r)};


	Color base = makeColor555(src_color_float);
	
	WriteDiff(dst, true);
	WriteFlip(dst, false);
	WriteColors555(dst, base, base);
	
	unsigned char best_tbl_idx = 0;
	unsigned char best_mod_idx = 0;
	unsigned int best_mod_err = 4294967295; 
	
	// Try all codeword tables to find the one giving the best results for this
	// block.
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		// Try all modifiers in the current table to find which one gives the
		// smallest error.
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];


			const Color color = makeColor(base, lum);
			
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

unsigned long compressBlock(global unsigned char* dst,
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
	
	// Compute the average color for each sub block and determine if differential
	// coding can be used.
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

   	for (size_t i=0; i<8; i++) 
    	dst[i] = (unsigned char)0;
	


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
	
__kernel void compress(__global uint8 *src,
						__global uint8 *dst,
						int width,
						int height) {
	

	int h = get_global_id(0);
	int w = get_global_id(1);

	Color ver_blocks[16];
	Color hor_blocks[16];
	
	unsigned long compressed_error = 0;
	
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			const Color* row0 = ((unsigned char)src + y * (width * 4 * 4) + x * 4 *4);
			const Color* row1 = row0 + width;
			const Color* row2 = row1 + width;
			const Color* row3 = row2 + width;
			
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
			
			compressed_error += compressBlock((unsigned char)(dst + x * 8), ver_blocks, hor_blocks, 4294967295);
		}
	}	
	return;
}>>>> file: texture_compress_maria.cpp
#include "compress.hpp"

#include 
#include 
#include 
#include 
#include 
#include 

#include "helper.hpp"

using namespace std;

/**
* Retrieve GPU device
*/
void gpu_find(cl_device_id &device)
{
	cl_platform_id platform;
	cl_uint platform_num = 0;
	cl_platform_id* platform_list = NULL;

	cl_uint device_num = 0;
	cl_device_id* device_list = NULL;
	int platform_index;

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
		
		if (strncmp("NVIDIA", (char *)attr_data, 6) == 0) {
			platform_index = platf;
		}

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

		/* get num of available OpenCL devices type GPU on the selected platform */
		if(clGetDeviceIDs(platform, 
			CL_DEVICE_TYPE_GPU, 0, NULL, &device_num) == CL_DEVICE_NOT_FOUND) {
			device_num = 0;
			continue;
		}

		device_list = new cl_device_id[device_num];
		DIE(device_list == NULL, "alloc devices");

		/* get all available OpenCL devices type GPU on the selected platform */
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

			/* select device based on cli arguments */
			if ((platf == platform_index) && (dev == 0)){
				device = device_list[dev];
				cout << " <--- SELECTED ";
			}

			cout << endl;
		}
	}

	delete[] platform_list;
	delete[] device_list;
}


TextureCompressor::TextureCompressor() { 	
	cl_device_id device;
	gpu_find(device);
	this->device = device;
} 	// constructor/Users/grigore.lupescu/Desktop/RESEARCH/asc/teme/tema3/2018/Tema3-schelet/src/compress.cpp
TextureCompressor::~TextureCompressor() { }	// destructor
	
unsigned long TextureCompressor::compress(const uint8_t* src,
									  uint8_t* dst,
									  int width,
									  int height)
{
	cl_int ret;
	cl_context context;
	cl_command_queue cmd_queue;
	string kernel_src;
	cl_kernel kernel;
	cl_program program;
	int size = width * height * 4;


	/* create a context for the device */
	context = clCreateContext(0, 1, &this->device, NULL, NULL, &ret);
	CL_ERR(ret);
	

	/* create a command queue for the device in the context */
	cmd_queue = clCreateCommandQueue(context, this->device, 0, &ret);
	CL_ERR(ret);


	/* allocate buffer on the DEVICE (GPU/VRAM) */
	cl_mem srcBuf = clCreateBuffer(context, CL_MEM_READ_ONLY,
		sizeof(cl_uint8) * size, NULL, &ret);
	CL_ERR(ret);
    cl_mem dstBuf = clCreateBuffer(context, CL_MEM_READ_WRITE,
    	sizeof(cl_uint8) * size / 8, NULL, &ret);
	CL_ERR(ret);
    

    /* copy into the buffers */
    ret = clEnqueueWriteBuffer(cmd_queue, srcBuf, CL_TRUE, 0,
    	sizeof(uint8_t) * size, src, 0, NULL, NULL);
    CL_ERR(ret);

	/* retrieve kernel source */
	read_kernel("kernel_maria.cl", kernel_src);
	const char* kernel_c_str = kernel_src.c_str();

	/* create kernel program from source */
	program = clCreateProgramWithSource(context, 1,
		  &kernel_c_str, NULL, &ret);
	CL_ERR(ret);

	/* compile the program for the given set of devices */
	ret = clBuildProgram(program, 1, &this->device, "", NULL, NULL);
	CL_COMPILE_ERR(ret, program, this->device);

	/* create kernel associated to compiled source kernel */
	kernel = clCreateKernel(program, "compress", &ret);
	CL_ERR(ret);

	/* set OpenCL kernel argument */
	CL_ERR(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&srcBuf));
	CL_ERR(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&dstBuf));
	CL_ERR(clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&width));
	CL_ERR(clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&height));

	size_t globalSize[2] = {height / 4, width / 4};
	ret = clEnqueueNDRangeKernel(cmd_queue, kernel, 2, NULL,
		  globalSize, 0, 0, NULL, NULL);
	CL_ERR(ret);

	/* copy the buffers back */


	ret = clEnqueueReadBuffer(cmd_queue, dstBuf, CL_TRUE, 0,
		  sizeof(uint8_t) * size / 8, dst, 0, NULL, NULL);
	CL_ERR(ret);

	/* wait for all enqueued operations to finish */
	CL_ERR( clFinish(cmd_queue) );

	/* free all resources related to GPU */
	CL_ERR( clReleaseMemObject(srcBuf) );
	CL_ERR( clReleaseMemObject(dstBuf) );
	CL_ERR( clReleaseCommandQueue(cmd_queue) );
	CL_ERR( clReleaseContext(context) );
	return 0;
}
