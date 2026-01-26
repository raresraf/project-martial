
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
>>>> file: texture_compress_device.cl
// definitia uniunii Color
typedef union color {
	struct BgraColorType {
		uchar b;
		uchar g;
		uchar r;
		uchar a;
	} channels;
	uchar components[4];
	uint bits;
} Color;

inline uchar uchar_clamp(uchar val, uchar min, uchar max) {
	return val  max ? max : val);
}

inline int int_clamp(int val, int min, int max) {
	return val  max ? max : val);
}

inline uchar round_to_5_bits(float val) {
	return uchar_clamp(val * 31.0f / 255.0f + 0.5f, 0, 31);
}

inline uchar round_to_4_bits(float val) {
	return uchar_clamp(val * 15.0f / 255.0f + 0.5f, 0, 15);
}



static __constant short g_codeword_tables[8][4] __attribute__((aligned(16))) = {
	{-8, -2, 2, 8},
	{-17, -5, 5, 17},
	{-29, -9, 9, 29},
	{-42, -13, 13, 42},
	{-60, -18, 18, 60},
	{-80, -24, 24, 80},
	{-106, -33, 33, 106},
	{-183, -47, 47, 183}};

static __constant uchar g_mod_to_pix[4] = {3, 2, 0, 1};

static __constant uchar g_idx_to_num[4][8] = {
	{0, 4, 1, 5, 2, 6, 3, 7},        // Vertical block 0.
	{8, 12, 9, 13, 10, 14, 11, 15},  // Vertical block 1.
	{0, 4, 8, 12, 1, 5, 9, 13},      // Horizontal block 0.
	{2, 6, 10, 14, 3, 7, 11, 15}     // Horizontal block 1.
};

inline Color makeColor(const Color base, short lum) {
	int b = (int)(base.channels.b) + lum;
	int g = (int)(base.channels.g) + lum;
	int r = (int)(base.channels.r) + lum;
	Color color;
	color.channels.b = (uchar)int_clamp(b, 0, 255);
	color.channels.g = (uchar)int_clamp(g, 0, 255);
	color.channels.r = (uchar)int_clamp(r, 0, 255);
	return color;
}

inline uint getColorError(const Color u, const Color v) {
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

inline void WriteColors444(__global uchar* block,
						   const Color color0,
						   const Color color1) {
	// Write output color for BGRA textures.
	block[0] = (color0.channels.r & 0xf0) | (color1.channels.r >> 4);
	block[1] = (color0.channels.g & 0xf0) | (color1.channels.g >> 4);
	block[2] = (color0.channels.b & 0xf0) | (color1.channels.b >> 4);
}

inline void WriteColors555(__global uchar* block,
						   const Color color0,
						   const Color color1) {
	// Table for conversion to 3-bit two complement format.
	const uchar two_compl_trans_table[8] = {
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

inline void WriteCodewordTable(__global uchar* block,
							   uchar sub_block_id,
							   uchar table) {
	
	uchar shift = (2 + (3 - sub_block_id * 3));
	block[3] &= ~(0x07 << shift);
	block[3] |= table << shift;
}

inline void WritePixelData(__global uchar* block, uint pixel_data) {
	block[4] |= pixel_data >> 24;
	block[5] |= (pixel_data >> 16) & 0xff;
	block[6] |= (pixel_data >> 8) & 0xff;
	block[7] |= pixel_data & 0xff;
}

inline void WriteFlip(__global uchar* block, bool flip) {
	block[3] &= ~0x01;
	block[3] |= (uchar)(flip);
}

inline void WriteDiff(__global uchar* block, bool diff) {
	block[3] &= ~0x02;
	block[3] |= (uchar)(diff) << 1;
}

// Compress and rounds BGR888 into BGR444. The resulting BGR444 color is
// expanded to BGR888 as it would be in hardware after decompression. The
// actual 444-bit data is available in the four most significant bits of each
// channel.
inline Color makeColor444(const float* bgr) {
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

// Compress and rounds BGR888 into BGR555. The resulting BGR555 color is
// expanded to BGR888 as it would be in hardware after decompression. The
// actual 555-bit data is available in the five most significant bits of each
// channel.
inline Color makeColor555(const float* bgr) {
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
	
void getAverageColor(const Color* src, float* avg_color)
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

unsigned long computeLuminance(__global uchar* block,
						   const Color* src,
						   const Color base,
						   int sub_block_id,
						   __constant uchar* idx_to_num_tab,
						   unsigned long threshold)
{
	uint best_tbl_err = threshold;
	uchar best_tbl_idx = 0;
	uchar best_mod_idx[8][8];  // [table][texel]

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
		
		uint tbl_err = 0;
		
		for (unsigned int i = 0; i < 8; ++i) {
			// Try all modifiers in the current table to find which one gives the
			// smallest error.
			uint best_mod_err = threshold;
			for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
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

	uint pix_data = 0;

	for (unsigned int i = 0; i < 8; ++i) {
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

/**
 * Tries to compress the block under the assumption that it's a single color
 * block. If it's not the function will bail out without writing anything to
 * the destination buffer.
 */
bool tryCompressSolidBlock(__global uchar* dst,
						   const Color* src,
						   unsigned long* error)
{
	for (unsigned int i = 1; i < 16; ++i) {
		if (src[i].bits != src[0].bits)
			return false;
	}
	
	// Clear destination buffer so that we can "or" in the results.


	for (unsigned int i = 0; i < 8; i++)
		dst[i] = 0;

	float src_color_float[3] = {(float)(src->channels.b),
		(float)(src->channels.g),
		(float)(src->channels.r)};
	Color base = makeColor555(src_color_float);
	


	WriteDiff(dst, true);
	WriteFlip(dst, false);
	WriteColors555(dst, base, base);
	
	uchar best_tbl_idx = 0;
	uchar best_mod_idx = 0;
	uint best_mod_err = 0xFFFFFFFF; 
	
	// Try all codeword tables to find the one giving the best results for this
	// block.
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		// Try all modifiers in the current table to find which one gives the
		// smallest error.
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
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

void compressBlock(__global uchar* dst,
					const Color* ver_src,
					const Color* hor_src,
					unsigned long threshold)
{
	unsigned long solid_error = 0;
	if (tryCompressSolidBlock(dst, ver_src, &solid_error)) {
		return;
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
	uint sub_block_err[4] = {0};
	for (unsigned int i = 0; i < 4; ++i) {
		for (unsigned int j = 0; j < 8; ++j) {
			sub_block_err[i] += getColorError(sub_block_avg[i], sub_block_src[i][j]);
		}
	}
	
	bool flip =
	sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1];
	
	// Clear destination buffer so that we can "or" in the results.


	for (unsigned int i = 0; i < 8; i++)
		dst[i] = 0;
	
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
	
	// Compute luminance for the first sub block.
	computeLuminance(dst, sub_block_src[sub_block_off_0],
								   sub_block_avg[sub_block_off_0], 0,
								   g_idx_to_num[sub_block_off_0],
								   threshold);
	// Compute luminance for the second sub block.
	computeLuminance(dst, sub_block_src[sub_block_off_1],
								   sub_block_avg[sub_block_off_1], 1,
								   g_idx_to_num[sub_block_off_1],
								   threshold);
}

// Functia apelata pentru a fi realizata de kernel
__kernel void compress(const __global uchar* src,
						__global uchar* dst,
						const int width,
						const int height)
{
	// se iau x din impartirea width si y din impartirea height
	int x = get_global_id(0);
	int y = get_global_id(1);

	// se calculeaza deplasamentul in src si dst conform
	// deplasarii din cele 2 for-uri
	int depl_src = y * width * 16 + x * 16;
	int depl_dst = 8 * y * width / 4 + x * 8;

	int i, j;

	// definitia block-urilor si a row-urilor
	Color ver_blocks[16];
	Color hor_blocks[16];
	Color row0[4], row1[4], row2[4], row3[4];

	// copierea componentelor din sursa in row0 pornind din
	// deplasament, camp cu camp, echivalent aici cu byte cu byte
	for (i = 0; i < 4; i++) {
		row0[i].channels.b = *(src + depl_src + 4 * i);
		row0[i].channels.g = *(src + depl_src + 4 * i + 1);
		row0[i].channels.r = *(src + depl_src + 4 * i + 2);
		row0[i].channels.a = *(src + depl_src + 4 * i + 3);
	}

	// copierea componentelor din sursa in row1 pornind de pe
	// randul urmator, camp cu camp, echivalent aici cu byte cu byte
	for (i = 0; i < 4; i++) {
		row1[i].channels.b = *(src + depl_src + 4 * width + 4 * i);
		row1[i].channels.g = *(src + depl_src + 4 * width + 4 * i + 1);
		row1[i].channels.r = *(src + depl_src + 4 * width + 4 * i + 2);
		row1[i].channels.a = *(src + depl_src + 4 * width + 4 * i + 3);
	}

	// copierea componentelor din sursa in row2 pornind de pe
	// randul urmator, camp cu camp, echivalent aici cu byte cu byte
	for (i = 0; i < 4; i++) {
		row2[i].channels.b = *(src + depl_src + 8 * width + 4 * i);
		row2[i].channels.g = *(src + depl_src + 8 * width + 4 * i + 1);
		row2[i].channels.r = *(src + depl_src + 8 * width + 4 * i + 2);
		row2[i].channels.a = *(src + depl_src + 8 * width + 4 * i + 3);
	}

	// copierea componentelor din sursa in row3 pornind de pe
	// randul urmator, camp cu camp, echivalent aici cu byte cu byte
	for (i = 0; i < 4; i++) {
		row3[i].channels.b = *(src + depl_src + 12 * width + 4 * i);
		row3[i].channels.g = *(src + depl_src + 12 * width + 4 * i + 1);
		row3[i].channels.r = *(src + depl_src + 12 * width + 4 * i + 2);
		row3[i].channels.a = *(src + depl_src + 12 * width + 4 * i + 3);
	}

	// copierea din rows in blocks, conforma cu cea din sursa,
	// tot camp cu camp
	for (i = 0; i < 4; i++) {
		if (i < 2) {


			ver_blocks[i].channels.b = row0[i].channels.b;
			ver_blocks[i].channels.g = row0[i].channels.g;
			ver_blocks[i].channels.r = row0[i].channels.r;
			ver_blocks[i].channels.a = row0[i].channels.a;
		}
		else {
			ver_blocks[i + 6].channels.b = row0[i].channels.b;
			ver_blocks[i + 6].channels.g = row0[i].channels.g;
			ver_blocks[i + 6].channels.r = row0[i].channels.r;


			ver_blocks[i + 6].channels.a = row0[i].channels.a;
		}

		hor_blocks[i].channels.b = row0[i].channels.b;
		hor_blocks[i].channels.g = row0[i].channels.g;
		hor_blocks[i].channels.r = row0[i].channels.r;
		hor_blocks[i].channels.a = row0[i].channels.a;
	}

	for (i = 0; i < 4; i++) {
		if (i < 2) {
			ver_blocks[i + 2].channels.b = row1[i].channels.b;
			ver_blocks[i + 2].channels.g = row1[i].channels.g;
			ver_blocks[i + 2].channels.r = row1[i].channels.r;
			ver_blocks[i + 2].channels.a = row1[i].channels.a;
		}
		else {
			ver_blocks[i + 8].channels.b = row1[i].channels.b;
			ver_blocks[i + 8].channels.g = row1[i].channels.g;
			ver_blocks[i + 8].channels.r = row1[i].channels.r;
			ver_blocks[i + 8].channels.a = row1[i].channels.a;
		}

		hor_blocks[i + 4].channels.b = row1[i].channels.b;
		hor_blocks[i + 4].channels.g = row1[i].channels.g;
		hor_blocks[i + 4].channels.r = row1[i].channels.r;
		hor_blocks[i + 4].channels.a = row1[i].channels.a;
	}

	for (i = 0; i < 4; i++) {
		if (i < 2) {
			ver_blocks[i + 4].channels.b = row2[i].channels.b;
			ver_blocks[i + 4].channels.g = row2[i].channels.g;
			ver_blocks[i + 4].channels.r = row2[i].channels.r;
			ver_blocks[i + 4].channels.a = row2[i].channels.a;
		}
		else {
			ver_blocks[i + 10].channels.b = row2[i].channels.b;
			ver_blocks[i + 10].channels.g = row2[i].channels.g;
			ver_blocks[i + 10].channels.r = row2[i].channels.r;
			ver_blocks[i + 10].channels.a = row2[i].channels.a;
		}

		hor_blocks[i + 8].channels.b = row2[i].channels.b;
		hor_blocks[i + 8].channels.g = row2[i].channels.g;
		hor_blocks[i + 8].channels.r = row2[i].channels.r;
		hor_blocks[i + 8].channels.a = row2[i].channels.a;
	}

	for (i = 0; i < 4; i++) {
		if (i < 2) {
			ver_blocks[i + 6].channels.b = row3[i].channels.b;
			ver_blocks[i + 6].channels.g = row3[i].channels.g;
			ver_blocks[i + 6].channels.r = row3[i].channels.r;
			ver_blocks[i + 6].channels.a = row3[i].channels.a;
		}
		else {
			ver_blocks[i + 12].channels.b = row3[i].channels.b;
			ver_blocks[i + 12].channels.g = row3[i].channels.g;
			ver_blocks[i + 12].channels.r = row3[i].channels.r;
			ver_blocks[i + 12].channels.a = row3[i].channels.a;
		}

		hor_blocks[i + 12].channels.b = row3[i].channels.b;
		hor_blocks[i + 12].channels.g = row3[i].channels.g;
		hor_blocks[i + 12].channels.r = row3[i].channels.r;
		hor_blocks[i + 12].channels.a = row3[i].channels.a;
	}

	// apelarea functiei compressBlock, cu block-ul din destinatie
	// la care se adauga deplasamentul conform algoritmului,
	// block-urile vertical si orizontal si valoarea maxima INT
	compressBlock(dst + depl_dst, ver_blocks, hor_blocks, 0x7FFFFFFF);
}
>>>> file: texture_compress_skl.cpp
#include "compress.hpp"
#include "helper.hpp"

#include 
#include 
#include 
#include 
#include 

using namespace std;

TextureCompressor::TextureCompressor()
{	
	int selected = 0;
	cl_uint device_num = 0;
	cl_uint platform_num = 0;
	cl_platform_id platform;

	// selectia numarului de platforme si alocarea vectorului
	CL_ERR( clGetPlatformIDs(0, NULL, &platform_num));
	platform_ids = new cl_platform_id[platform_num];
	DIE(platform_ids == NULL, "alloc platform_list");

	// popularea vectorului cu platforme
	CL_ERR( clGetPlatformIDs(platform_num, platform_ids, NULL));

	// iterarea prin platforme
	for (int platf = 0; platf < platform_num; platf++) {
		// selectia platformei cu indexul platf
		platform = platform_ids[platf];

		// selectia numarului de device-uri GPU din platforma
		clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &device_num);

		// daca exista macar un device GPU
		if (device_num > 0) {
			// alocarea vectorului de device-uri
			device_ids = new cl_device_id[device_num];
			DIE(device_ids == NULL, "alloc devices");

			// popularea vectorului cu device-urile GPU din platforma
			CL_ERR( clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU,
									   device_num, device_ids, NULL));

			// selectia primului device existent
			for (int dev = 0; dev < device_num; dev++) {
				device = device_ids[dev];
				selected = 1;
				break;
			}
		}

		// daca s-a selectat un device, se iese din bucla
		if (selected == 1)
			break;
	}
}

TextureCompressor::~TextureCompressor()
{
	// eliberarea vectorilor


	delete[] platform_ids;
	delete[] device_ids;
}
	
unsigned long TextureCompressor::compress(const uint8_t* src,
									  uint8_t* dst,
									  int width,
									  int height)
{
	cl_mem src_in;
	cl_mem dst_out;
	int ret;
	string kernel_src;

	// crearea contextului pe device-ul selectat
	context = clCreateContext(0, 1, &device, NULL, NULL, &ret);
	CL_ERR( ret );

	// crearea cozii pentru contextul selectat si device
	command_queue = clCreateCommandQueue(context, device, 0, &ret);
	CL_ERR( ret );

	// preluarea sursei kernel-ului
	read_kernel("texture_compress_device.cl", kernel_src);
	const char* kernel_c_str = kernel_src.c_str();

	// crearea programului cu sursa kernel-ului
	program = clCreateProgramWithSource(context, 1,
				&kernel_c_str, NULL, &ret);
	CL_ERR( ret );

	// constructia programului
	ret = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	CL_COMPILE_ERR(ret, program, device);

	// crearea kernel-ului propriu-zis, cu functia de apelat
	kernel = clCreateKernel(program, "compress", &ret);
	CL_ERR( ret );
	
	// crearea buffer-ului pentru sursa imaginii cu dimensiunea din compress.cpp
	src_in = clCreateBuffer(context,  CL_MEM_READ_ONLY,
				width * height * 4, NULL, NULL);
	DIE(src_in == 0, "alloc src_in");
	// crearea buffer-ului pentru imaginea prelucrata cu dimensiunea din compress.cpp
	dst_out = clCreateBuffer(context,  CL_MEM_READ_WRITE,
				width * height * 4 / 8, NULL, NULL);
	DIE(dst_out == 0, "alloc dst_out");

	// scrierea in buffer a imaginii-sursa,
	ret = clEnqueueWriteBuffer(command_queue, src_in, CL_TRUE, 0,
							   width * height * 4, src, 0, NULL, NULL);
	CL_ERR(ret);
	// scrierea in buffer-ul imaginii prelucrate din dst (0)
	ret = clEnqueueWriteBuffer(command_queue, dst_out, CL_TRUE, 0,
							   width * height * 4 / 8, dst, 0, NULL, NULL);
	CL_ERR(ret);

	// trimiterea argumentelor la functia "compress" din kernel
	CL_ERR(clSetKernelArg(kernel, 0, sizeof(cl_mem), &src_in));
	CL_ERR(clSetKernelArg(kernel, 1, sizeof(cl_mem), &dst_out));
	CL_ERR(clSetKernelArg(kernel, 2, sizeof(int), &width));
	CL_ERR(clSetKernelArg(kernel, 3, sizeof(int), &height));

	// setarea dimensiunilor imaginii, avand in vedere ca se prelucreaza
	// blocuri de cate 16 pixeli deodata
	size_t globalSize[2] = {(size_t)(width / 4), (size_t)(height / 4)};

	// comanda de executie kernel-ului
	CL_ERR(clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, 
								globalSize, 0, 0, NULL, NULL));

	// asteptarea executiei comenzii din coada
	clFinish(command_queue);

	// citirea imaginii destinatie prelucrata in kernel
	CL_ERR(clEnqueueReadBuffer(command_queue, dst_out, CL_TRUE, 0,
						width * height * 4 / 8, dst, 0, NULL, NULL));

	// eliberarea resurselor
	clReleaseMemObject(src_in);
	clReleaseMemObject(dst_out);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);

	return 0;
}
