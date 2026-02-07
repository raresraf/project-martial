
>>>> file: helper.cpp
#include 
#include 
#include 
#include 
#include 

#include "helper.hpp"

using namespace std;

/**
 * @brief Checks the return code of an OpenCL function and prints an error message if it indicates failure.
 * @param cl_ret The integer return code from an OpenCL API call.
 * @return `1` if the OpenCL function failed, `0` otherwise.
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
 * @brief Checks the return code of an OpenCL program compilation and prints an error log if it indicates failure.
 * @param cl_ret The integer return code from `clBuildProgram`.
 * @param program The `cl_program` object that was being built.
 * @param device The `cl_device_id` on which the program was compiled.
 * @return `1` if the OpenCL program compilation failed, `0` otherwise.
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
 * @brief Reads the content of an OpenCL kernel file into a string.
 * @param file_name The name of the kernel file to read.
 * @param str_kernel A reference to a string where the kernel source will be stored.
 * Pre-condition: The file specified by `file_name` must exist and be accessible in the same directory as the binary.
 * Post-condition: `str_kernel` contains the entire content of the kernel file.
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
 * @brief Converts an OpenCL error code into a human-readable string.
 * This function is a lookup table for various `cl_int` error codes.
 * @param err The `cl_int` OpenCL error code.
 * @return A C-style string describing the error.
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
 * @brief Retrieves and prints the OpenCL program build log, typically used for debugging compilation errors.
 * @param program The `cl_program` object for which to retrieve the build log.
 * @param device The `cl_device_id` on which the program was compiled.
 * Post-condition: The compiler build log is printed to standard output.
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
>>>> file: kernel_skl.cl
/* Tema 3 ASC - Compresie ETC1
 *
 * Popescu Ana-Cosmina,
 * 331CC
 */

typedef union Color {
	struct BgraColorType {
		uchar b;
		uchar g;
		uchar r;
		uchar a;
	} channels;
	uchar components[4];
	uint bits;
} Color;

/* Define a threshold for computation time. */
__constant int max_int = 30000;

/* Write clamp function that works with ints. */
inline int my_clamp(int val, int mmin, int mmax)
{
	return val  mmax ? mmax : val);
}

inline uchar round_to_5_bits(float val)
{
	int res = my_clamp(val * 31.0f / 255.0f + 0.5f, 0, 31);
	uchar converted_res = (uchar) res;
	return converted_res;
}


inline uchar round_to_4_bits(float val)
{
        int res = my_clamp(val * 15.0f / 255.0f + 0.5f, 0, 15);
        uchar converted_res = (uchar) res;
        return converted_res;
}

__attribute__((aligned(16))) __constant short g_codeword_tables[8][4] = {
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
	//{0, 4, 1, 5, 2, 6, 3, 7},        // Vertical block 0.
	//{8, 12, 9, 13, 10, 14, 11, 15},  // Vertical block 1.
	{0, 2, 4, 6, 1, 3, 5, 7},
	{8, 10, 12, 14, 9, 11, 13, 15},
	{0, 4, 8, 12, 1, 5, 9, 13},      // Horizontal block 0.
	{2, 6, 10, 14, 3, 7, 11, 15}     // Horizontal block 1.
};

/* Table for conversion to 3-bit two complement format. */
__constant uchar two_compl_trans_table[8] = {
	4,  // -4 (100b)
	5,  // -3 (101b)
	6,  // -2 (110b)
	7,  // -1 (111b)
	0,  //  0 (000b)
	1,  //  1 (001b)
	2,  //  2 (010b)
	3,  //  3 (011b)
};

/* Constructs a color from a given base color and luminance value. */
inline Color makeColor(Color* base, short lum)
{
	int b = ((int) base->channels.b) + lum;
	int g = ((int) base->channels.g) + lum;
	int r = ((int) base->channels.r) + lum;
	Color color;
	color.channels.b = (uchar) my_clamp(b, 0, 255);


	color.channels.g = (uchar) my_clamp(g, 0, 255);
	color.channels.r = (uchar) my_clamp(r, 0, 255);
	return color;
}

/* Calculates the error metric for two colors. A small error signals that the
 * colors are similar to each other, a large error the signals the opposite.
 */
inline uint getColorError(Color u, Color v)
{
	int delta_b = (int) (u.channels.b - v.channels.b);
	int delta_g = (int) (u.channels.g - v.channels.g);
	int delta_r = (int) (u.channels.r - v.channels.r);
	return delta_b * delta_b + delta_g * delta_g + delta_r * delta_r;
}

inline void WriteColors444(__global uchar* block, int offset, 
						Color* color0, Color* color1)
{
	/* Write output color for BGRA textures. */
	block[offset + 0] =
		(color0->channels.r & 0xf0) | (color1->channels.r >> 4);
	block[offset + 1] =
		(color0->channels.g & 0xf0) | (color1->channels.g >> 4);
	block[offset + 2] =
		(color0->channels.b & 0xf0) | (color1->channels.b >> 4);
}

inline void WriteColors555(__global uchar* block, int offset,
						Color* color0, Color* color1)
{
	short delta_r = 
		(short) (color1->channels.r >> 3) - (color0->channels.r >> 3);
	short delta_g =
		(short) (color1->channels.g >> 3) - (color0->channels.g >> 3);
	short delta_b =
		(short) (color1->channels.b >> 3) - (color0->channels.b >> 3);
	
	/* Write output color for BGRA textures. */
	block[offset + 0] = (color0->channels.r & 0xf8) |
					two_compl_trans_table[delta_r + 4];
	block[offset + 1] = (color0->channels.g & 0xf8) |
					two_compl_trans_table[delta_g + 4];
	block[offset + 2] = (color0->channels.b & 0xf8) |
					two_compl_trans_table[delta_b + 4];
}

inline void WriteCodewordTable(__global uchar* block, int offset,
					uchar sub_block_id, uchar table)
{
	uchar shift = (2 + (3 - sub_block_id * 3));
	block[offset + 3] &= ~(0x07 << shift);
	block[offset + 3] |= table << shift;
}

inline void WritePixelData(__global uchar* block, int offset, uint pixel_data)
{
	block[offset + 4] |= pixel_data >> 24;
	block[offset + 5] |= (pixel_data >> 16) & 0xff;
	block[offset + 6] |= (pixel_data >> 8) & 0xff;
	block[offset + 7] |= pixel_data & 0xff;
}

inline void WriteFlip(__global uchar* block, int offset, bool flip)
{
	block[offset + 3] &= ~0x01;
	block[offset + 3] |= (uchar) flip;
}

inline void WriteDiff(__global uchar* block, int offset, bool diff)
{
	block[offset + 3] &= ~0x02;
	block[offset + 3] |= (uchar) diff << 1;
}

/* Compress and rounds BGR888 into BGR444. The resulting BGR444 color is
 * expanded to BGR888 as it would be in hardware after decompression. The
 * actual 444-bit data is available in the four most significant bits of each
 * channel.
 */
inline Color makeColor444(float* bgr)
{
	uchar b4 = round_to_4_bits(bgr[0]);
	uchar g4 = round_to_4_bits(bgr[1]);
	uchar r4 = round_to_4_bits(bgr[2]);
	
	Color bgr444;
	bgr444.channels.b = (b4 << 4) | b4;
	bgr444.channels.g = (g4 << 4) | g4;
	bgr444.channels.r = (r4 << 4) | r4;
	/* Added to distinguish between expanded 555 and 444 colors. */
	bgr444.channels.a = 0x44;

	return bgr444;
}

/* Compress and rounds BGR888 into BGR555. The resulting BGR555 color is
 * expanded to BGR888 as it would be in hardware after decompression. The
 * actual 555-bit data is available in the five most significant bits of each
 * channel.
 */
inline Color makeColor555(const float* bgr)
{
	uchar b5 = round_to_5_bits(bgr[0]);
	uchar g5 = round_to_5_bits(bgr[1]);
	uchar r5 = round_to_5_bits(bgr[2]);

	Color bgr555;
	bgr555.channels.b = (b5 > 2);
	bgr555.channels.g = (g5 > 2);


	bgr555.channels.r = (r5 > 2);
	/* Added to distinguish between expanded 555 and 444 colors. */
	bgr555.channels.a = 0x55;

	return bgr555;
}

void getAverageColor(Color* src, float* avg_color)
{
	uint sum_b = 0, sum_g = 0, sum_r = 0;
	
	for (uint i = 0; i < 8; ++i) {
		sum_b += src[i].channels.b;


		sum_g += src[i].channels.g;
		sum_r += src[i].channels.r;
	}
	
	float kInv8 = 1.0f / 8.0f;
	avg_color[0] = (float) (sum_b) * kInv8;
	avg_color[1] = (float) (sum_g) * kInv8;
	avg_color[2] = (float) (sum_r) * kInv8;
}

/* Tries to compress the block under the assumption that it's a single color
 * block. If it's not the function will bail out without writing anything to
 * the destination buffer.
 */
bool tryCompressSolidBlock(__global uchar* dst, int offset, Color* src,
			   					ulong* error)
{
	for (int i = 1; i < 16; ++i) {
		if (src[i].bits != src[0].bits)
			return false;
	}
	
	/* Clear destination buffer so that we can "or" in the results. */
	 for (uint i = 0; i < 7; i++)
                dst[offset + i] = 0;
	
	float src_color_float[3] = { (float) (src->channels.b),
				     (float) (src->channels.g),
				     (float) (src->channels.r)};
	Color base = makeColor555(src_color_float);
	
	WriteDiff(dst, offset, true);
	WriteFlip(dst, offset, false);


	WriteColors555(dst, offset, &base, &base);
	
	uchar best_tbl_idx = 0;
	uchar best_mod_idx = 0;
	uint best_mod_err = max_int; 
	
	/* Try all codeword tables to find the one giving the best results for
	 * this block.
	 */
	for (uint tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		/* Try all modifiers in the current table to find which one
		 * gives the smallest error.
		 */
		for (uint mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			Color color = makeColor(&base, lum);
			
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
	
	WriteCodewordTable(dst, offset, 0, best_tbl_idx);
	WriteCodewordTable(dst, offset, 1, best_tbl_idx);
	
	uchar pix_idx = g_mod_to_pix[best_mod_idx];
	uint lsb = pix_idx & 0x1;
	uint msb = pix_idx >> 1;
	
	uint pix_data = 0;
	for (uint i = 0; i < 2; ++i) {
		for (uint j = 0; j < 8; ++j) {
			/* Obtain the texel number as specified in the
			 * standard.
			 */
			int texel_num = g_idx_to_num[i][j];
			pix_data |= msb << (texel_num + 16);
			pix_data |= lsb << (texel_num);
		}
	}
	
	WritePixelData(dst, offset, pix_data);
	*error = 16 * best_mod_err;
	return true;
}

ulong computeLuminance(__global uchar* block, int offset, Color* src,


		       Color* base, int sub_block_id,
		       __constant uchar* idx_to_num_tab, ulong threshold)
{
	uint best_tbl_err = threshold;
	uchar best_tbl_idx = 0;
	uchar best_mod_idx[8][8];

	/* Try all codeword tables to find the one giving the best results for
	 * this block.
	 */
	for (uint tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		/* Pre-compute all the candidate colors; combinations of the
		 * base color and all available luminance values.
		 */
		Color candidate_color[4];  // [modifier]
		for (uint mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			candidate_color[mod_idx] = makeColor(base, lum);
		}
		
		uint tbl_err = 0;
		
		for (uint i = 0; i < 8; ++i) {
			/* Try all modifiers in the current table to find which
			 * one gives the smallest error.
			 */
			uint best_mod_err = threshold;


			for (uint mod_idx = 0; mod_idx < 4; ++mod_idx) {
				Color color = candidate_color[mod_idx];
				
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

	WriteCodewordTable(block, offset, sub_block_id, best_tbl_idx);

	uint pix_data = 0;

	for (uint i = 0; i < 8; ++i) {
		uchar mod_idx = best_mod_idx[best_tbl_idx][i];
		uchar pix_idx = g_mod_to_pix[mod_idx];
		
		uint lsb = pix_idx & 0x1;
		uint msb = pix_idx >> 1;
		
		/* Obtain the texel number as specified in the standard. */
		int texel_num = idx_to_num_tab[i];
		pix_data |= msb << (texel_num + 16);
		pix_data |= lsb << (texel_num);
	}

	WritePixelData(block, offset, pix_data);

	return best_tbl_err;
}

ulong compressBlock(__global uchar* dst, int dst_offset, Color* ver_src,
					Color* hor_src, ulong threshold)
{	
	/* Can be used for further improvements.
	ulong solid_error = 0;
	if (tryCompressSolidBlock(dst, dst_offset, ver_src, &solid_error)) {
		return solid_error;
	} */

	Color* sub_block_src[4] = {ver_src, ver_src + 8, hor_src, hor_src + 8};
	
	Color sub_block_avg[4];
	bool use_differential[2] = {true, true};
	
	/* Compute the average color for each sub block and determine if
	 * differential coding can be used.
	 */
	for (uint i = 0, j = 1; i < 4; i += 2, j += 2) {
		float avg_color_0[3];
		getAverageColor(sub_block_src[i], avg_color_0);
		Color avg_color_555_0 = makeColor555(avg_color_0);
		
		float avg_color_1[3];
		getAverageColor(sub_block_src[j], avg_color_1);
		Color avg_color_555_1 = makeColor555(avg_color_1);
		
		for (uint light_idx = 0; light_idx < 3; ++light_idx) {
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
	
	/* Compute the error of each sub block before adjusting for luminance.
	 * These error values are later used for determining if we should flip
	 * the sub-block or not.
	 */
	uint sub_block_err[4] = {0};
	for (uint i = 0; i < 4; ++i) {
		for (uint j = 0; j < 8; ++j) {
			sub_block_err[i] += getColorError(sub_block_avg[i],
							  sub_block_src[i][j]);
		}
	}
	
	bool flip = sub_block_err[2] + sub_block_err[3] <
					sub_block_err[0] + sub_block_err[1];
	
	/* Clear destination buffer so that we can "or" in the results. */
	for (uint i = 0; i < 7; i++)
		dst[dst_offset + i] = 0;
	
	WriteDiff(dst, dst_offset, use_differential[!!flip]);
	WriteFlip(dst, dst_offset, flip);
	
	uchar sub_block_off_0 = flip ? 2 : 0;
	uchar sub_block_off_1 = sub_block_off_0 + 1;
	
	if (use_differential[!!flip]) {
		WriteColors555(dst,
			       dst_offset,
			       &sub_block_avg[sub_block_off_0],
			       &sub_block_avg[sub_block_off_1]);
	} else {
		WriteColors444(dst,
			       dst_offset,
			       &sub_block_avg[sub_block_off_0],
			       &sub_block_avg[sub_block_off_1]);
	}
	
	/* Can be used for further performance improvements.
	ulong lumi_error1 = 0, lumi_error2 = 0;
	
	// Compute luminance for the first sub block.
	lumi_error1 = computeLuminance(dst, dst_offset,
				       sub_block_src[sub_block_off_0],
				       &sub_block_avg[sub_block_off_0],
				       0,
				       g_idx_to_num[sub_block_off_0],
				       threshold);
	// Compute luminance for the second sub block.
	lumi_error2 = computeLuminance(dst, dst_offset,
				       sub_block_src[sub_block_off_1],
				       &sub_block_avg[sub_block_off_1],
				       1,
				       g_idx_to_num[sub_block_off_1],
				       threshold);
	return lumi_error1 + lumi_error2; */

	return 0;
}

/* Dumb memcpy function that only knows how to copy 1 element at a time. */
void my_memcpy(Color *blocks, __global uchar *src)
{
	uchar *bls = (uchar *) blocks;
	for (int i = 0; i < 4; i++)
		bls[i] = src[i];	
}

/* Compression kernel. */
__kernel void kernel_compress(__global uchar* src, __global uchar* dst,
							int width, int height)
{
	int gid_0 = get_global_id(0);
	int gid_1 = get_global_id(1);

	Color ver_blocks[16];
	Color hor_blocks[16];
	
	/* Compute source offset. */
	int x = gid_1 * width * 4 * 4 + 4 * 4 * gid_0;
	/* Compute destination offset. */
	int y = 8 * gid_1 * width / 4 + gid_0 * 8; 

	/* Copy vertical blocks. */
	my_memcpy(&ver_blocks[0], src + x);	
	my_memcpy(&ver_blocks[1], src + x + 4);
	my_memcpy(&ver_blocks[2], src + x + width);
	my_memcpy(&ver_blocks[3], src + x + width + 4);
	my_memcpy(&ver_blocks[4], src + x + 2 * width);
	my_memcpy(&ver_blocks[5], src + x + 2 * width + 4);
	my_memcpy(&ver_blocks[6], src + x + 3 * width);
	my_memcpy(&ver_blocks[7], src + x + 3 * width + 4);

	my_memcpy(&ver_blocks[8], src + x + 8);
	my_memcpy(&ver_blocks[9], src + x + 12);
	my_memcpy(&ver_blocks[10], src + x + width + 8);
	my_memcpy(&ver_blocks[11], src + x + width + 12);
	my_memcpy(&ver_blocks[12], src + x + 2 * width + 8);
	my_memcpy(&ver_blocks[13], src + x + 2 * width + 12);
	my_memcpy(&ver_blocks[14], src + x + 3 * width + 8);
	my_memcpy(&ver_blocks[15], src + x + 3 * width + 12);

	/* Copy horizontal blocks. */
	my_memcpy(&hor_blocks[0], src + x);
	my_memcpy(&hor_blocks[1], src + x + 4);
	my_memcpy(&hor_blocks[2], src + x + 8);
	my_memcpy(&hor_blocks[3], src + x + 12);
	my_memcpy(&hor_blocks[4], src + x + width);
	my_memcpy(&hor_blocks[5], src + x + width + 4);
	my_memcpy(&hor_blocks[6], src + x + width + 8);
	my_memcpy(&hor_blocks[7], src + x + width + 12);

	my_memcpy(&hor_blocks[8], src + x + 2 * width);
	my_memcpy(&hor_blocks[9], src + x + 2 * width + 4);
	my_memcpy(&hor_blocks[10], src + x + 2 * width + 8);
	my_memcpy(&hor_blocks[11], src + x + 2 * width + 12);
	my_memcpy(&hor_blocks[12], src + x + 3 * width);
	my_memcpy(&hor_blocks[13], src + x + 3 * width + 4);
	my_memcpy(&hor_blocks[14], src + x + 3 * width + 8);
	my_memcpy(&hor_blocks[15], src + x + 3 * width + 12);

	compressBlock(dst, y, ver_blocks, hor_blocks, max_int);
}
>>>> file: texture_compress_skl.cpp
/* Tema 3 ASC - Compresie ETC1
 *
 * Popescu Ana-Cosmina,
 * 331CC
 */

#include "compress.hpp"

#include 
#include 
#include 
#include 
#include 

#include "helper.hpp"

using namespace std;

/**
 *  * Retrieve GPU device
 *   */
void gpu_find(cl_device_id &device, uint platform_select, uint device_select)
{
	cl_platform_id platform;
	cl_uint platform_num = 0;
	cl_platform_id* platform_list = NULL;
	
	cl_uint device_num = 0;
	cl_device_id* device_list = NULL;
	
	size_t attr_size = 0;
	cl_char* attr_data = NULL;
	
	/* Get num of available OpenCL platforms. */
	CL_ERR( clGetPlatformIDs(0, NULL, &platform_num));
	platform_list = new cl_platform_id[platform_num];
	DIE(platform_list == NULL, "alloc platform_list");
	
	/* Get all available OpenCL platforms. */
	CL_ERR( clGetPlatformIDs(platform_num, platform_list, NULL));
	
	/* Find out all platforms and VENDOR/VERSION properties. */
	for(uint platf=0; platf<platform_num; platf++)
	{
		/* Get attribute CL_PLATFORM_VENDOR. */
		CL_ERR( clGetPlatformInfo(platform_list[platf],
				CL_PLATFORM_VENDOR, 0, NULL, &attr_size));
		attr_data = new cl_char[attr_size];
		DIE(attr_data == NULL, "alloc attr_data");
		
		/* Get data CL_PLATFORM_VENDOR. */
		CL_ERR( clGetPlatformInfo(platform_list[platf],
			CL_PLATFORM_VENDOR, attr_size, attr_data, NULL));
		delete[] attr_data;
		
		/* Get attribute size CL_PLATFORM_VERSION. */
		CL_ERR( clGetPlatformInfo(platform_list[platf],
				  CL_PLATFORM_VERSION, 0, NULL, &attr_size));
		attr_data = new cl_char[attr_size];
		DIE(attr_data == NULL, "alloc attr_data");
		
		/* Get data size CL_PLATFORM_VERSION. */
		CL_ERR( clGetPlatformInfo(platform_list[platf],
			  CL_PLATFORM_VERSION, attr_size, attr_data, NULL));
		delete[] attr_data;
		
		/* No valid platform found. */
		platform = platform_list[platf];
		DIE(platform == 0, "platform selection");
		
		/* Get num of available OpenCL devices type GPU on the
 		 * selected platform.
 		 */
		CL_ERR( clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL,
								&device_num));
		device_list = new cl_device_id[device_num];
		DIE(device_list == NULL, "alloc devices");
		
		/* Get all available OpenCL devices type GPU on the selected
		 * platform.
		 */
		CL_ERR( clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL,
					 device_num, device_list, NULL));
		
		/* Find out all devices and TYPE/VERSION properties. */
		for(uint dev=0; dev<device_num; dev++)
		{
			/* Get attribute size. */
			CL_ERR( clGetDeviceInfo(device_list[dev],
					CL_DEVICE_NAME, 0, NULL, &attr_size));
			attr_data = new cl_char[attr_size];
			DIE(attr_data == NULL, "alloc attr_data");
			
			/* Get attribute CL_DEVICE_NAME. */
			CL_ERR( clGetDeviceInfo(device_list[dev],
				CL_DEVICE_NAME, attr_size, attr_data, NULL));
			delete[] attr_data;
			
			/* Get attribute size. */
			CL_ERR( clGetDeviceInfo(device_list[dev],
				CL_DEVICE_VERSION, 0, NULL, &attr_size));
			attr_data = new cl_char[attr_size];
			DIE(attr_data == NULL, "alloc attr_data");
			
			/* Get attribute CL_DEVICE_VERSION. */
			CL_ERR( clGetDeviceInfo(device_list[dev],
						CL_DEVICE_VERSION,
						attr_size,
						attr_data,
						NULL));
			delete[] attr_data;
			
			if(dev == device_select)
				device = device_list[dev];
		}
	}
	
	delete[] platform_list;
	delete[] device_list;
}

/* Constructor. */
TextureCompressor::TextureCompressor() {
	int platform_select = 0;
	int device_select = 1;
	gpu_find(device, platform_select, device_select);
}

/* Destructor. */
TextureCompressor::~TextureCompressor() { }
	
unsigned long TextureCompressor::compress(const uint8_t* src, uint8_t* dst,
					  	int width, int height)
{
	/* TextureCompressor class has: device,
   	 * context, program, command_queue, kernel, device_ids, platform_ids.
   	 */
	cl_int ret;

	string kernel_src;
	int src_size = width * height * 4;
	int dst_size = width * height * 4 / 8;

	/* Create a context for the device. */
	context = clCreateContext(0, 1, &device, NULL, NULL, &ret);
	CL_ERR( ret );

	/* Create a command queue for the device in the context. */
	command_queue = clCreateCommandQueue(context, device,
					CL_QUEUE_PROFILING_ENABLE, &ret);
	CL_ERR( ret );

	/* Allocate buffer on the DEVICE (GPU). */
	cl_mem buf_src = clCreateBuffer(context, CL_MEM_READ_ONLY,
				sizeof(uint8_t) * src_size, NULL, &ret);
	CL_ERR( ret );
	DIE(buf_src == 0, "alloc buf_src");

	cl_mem buf_dst = clCreateBuffer(context, CL_MEM_READ_WRITE,
				sizeof(uint8_t) * dst_size, NULL, &ret);
	CL_ERR( ret );
	DIE(buf_dst == 0, "alloc_buf_dst");
	
	/* Copy src buffer to the GPU device. */
	CL_ERR( clEnqueueWriteBuffer(command_queue, buf_src, CL_TRUE, 0,
		  	sizeof(uint8_t) * src_size, src, 0, NULL, NULL));
	/* Retrieve kernel source. */
	read_kernel("kernel_skl.cl", kernel_src);
	const char* kernel_c_str = kernel_src.c_str();

	/* Create kernel program from source. */
	program = clCreateProgramWithSource(context, 1, &kernel_c_str, NULL,
					    				&ret);
	CL_ERR( ret );

	/* Compile the program for the given set of devices. */
	ret = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	CL_COMPILE_ERR( ret, program, device );

	/* Create kernel associated to compiled source kernel. */
	kernel = clCreateKernel(program, "kernel_compress", &ret);
	CL_ERR( ret );

	/* set OpenCL kernel argument */
	CL_ERR( clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&buf_src) );
	CL_ERR( clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&buf_dst) );
	CL_ERR( clSetKernelArg(kernel, 2, sizeof(cl_uint), (void *)&width) );
	CL_ERR( clSetKernelArg(kernel, 3, sizeof(cl_uint), (void *)&height) );

	/* Profile execution of OpenCL kernel. */
	cl_event event;
	size_t globalSize[2] = {(size_t) width / 4, (size_t) height / 4};
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL,
					globalSize, 0, 0, NULL, &event);
	CL_ERR( ret );
	CL_ERR( clWaitForEvents(1, &event));

	/* Copy the buffers back. */
	CL_ERR( clEnqueueReadBuffer(command_queue, buf_dst, CL_TRUE, 0,
			sizeof(uint8_t) * dst_size, dst, 0, NULL, NULL));

	/* Wait for all enqueued operations to finish. */
	CL_ERR( clFinish(command_queue) );

	/* Free all resources related to GPU. */
	CL_ERR( clReleaseMemObject(buf_src) );
	CL_ERR( clReleaseMemObject(buf_dst) );

	return 0;
}
