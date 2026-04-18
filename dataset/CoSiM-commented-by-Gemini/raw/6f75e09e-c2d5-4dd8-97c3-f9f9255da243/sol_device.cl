/**
 * @file sol_device.cl
 * @brief An OpenCL implementation for a block-based texture compression algorithm.
 *
 * This file contains both the OpenCL device code (kernels and helper functions)
 * for texture compression and the C++ host code for orchestrating the process.
 * The algorithm compresses 4x4 pixel blocks into a fixed-size output, using
 * techniques similar to the Ericsson Texture Compression (ETC) standard,
 * including differential color encoding and luminance modulation via codeword tables.
 */

// --- OpenCL Device Code ---

inline uchar my_clamp(uchar val, uchar min, uchar max) {
	return val < min ? min : (val > max ? max : val);
}

// Quantizes a 0-255 float value to 5 bits.
inline uchar round_to_5_bits(float val) {
	return my_clamp(val * 31.0f / 255.0f + 0.5f, 0, 31);
}
// Quantizes a 0-255 float value to 4 bits.
inline uchar round_to_4_bits(float val) {
	return my_clamp(val * 15.0f / 255.0f + 0.5f, 0, 15);
}

// Represents a 32-bit BGRA color, allowing access by channels, components, or raw bits.
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


/**
 * @brief Codeword tables for luminance modulation.
 * Stored in __constant memory for fast, cached access by all work-items.
 * Each table provides four intensity modifiers for a sub-block's base color.
 */
__attribute__((aligned(16))) static constant short g_codeword_tables[8][4] = {
	{-8, -2, 2, 8},
	{-17, -5, 5, 17},
	{-29, -9, 9, 29},
	{-42, -13, 13, 42},
	{-60, -18, 18, 60},
	{-80, -24, 24, 80},
	{-106, -33, 33, 106},
	{-183, -47, 47, 183}};


// Maps a 2-bit modifier index to a 2-bit pixel index.
static constant uchar g_mod_to_pix[4] = {3, 2, 0, 1};

// Tables for reordering pixel indices based on the block 'flip' mode,
// which determines how a 4x4 block is partitioned.
static constant uchar g_idx_to_num[4][8] = {
	{0, 4, 1, 5, 2, 6, 3, 7},
	{8, 12, 9, 13, 10, 14, 11, 15},
	{0, 4, 8, 12, 1, 5, 9, 13},
	{2, 6, 10, 14, 3, 7, 11, 15}
};


// Applies a luminance offset to a base color to create a new color.
inline union Color makeColor( union Color* base, short lum) {
	int b = (int)(base->channels.b) + lum;
	int g = (int)(base->channels.g) + lum;
	int r = (int)(base->channels.r) + lum;
	union Color color;
	color.channels.b = (uchar)(clamp(b, 0, 255));
	color.channels.g = (uchar)(clamp(g, 0, 255));
	color.channels.r = (uchar)(clamp(r, 0, 255));
	return color;
}

// Calculates the squared Euclidean distance between two colors.
inline uint getColorError( union Color* u,  union Color* v) {
	int delta_b = (int)(u->channels.b) - v->channels.b;
	int delta_g = (int)(u->channels.g) - v->channels.g;
	int delta_r = (int)(u->channels.r) - v->channels.r;
	return delta_b * delta_b + delta_g * delta_g + delta_r * delta_r;
}

// Writes two 4-bit-per-channel colors to the compressed block.
inline void WriteColors444(uchar* block,
						    union Color* color0,
						    union Color* color1) {
	block[0] = (color0->channels.r & 0xf0) | (color1->channels.r >> 4);
	block[1] = (color0->channels.g & 0xf0) | (color1->channels.g >> 4);
	block[2] = (color0->channels.b & 0xf0) | (color1->channels.b >> 4);
}

// Writes two 5-bit-per-channel colors using differential encoding.
inline void WriteColors555(uchar* block,
						    union Color* color0,
						    union Color* color1) {
	const uchar two_compl_trans_table[8] = { 4, 5, 6, 7, 0, 1, 2, 3 };
	short delta_r = (short)(color1->channels.r >> 3) - (color0->channels.r >> 3);
	short delta_g = (short)(color1->channels.g >> 3) - (color0->channels.g >> 3);
	short delta_b = (short)(color1->channels.b >> 3) - (color0->channels.b >> 3);
	
	block[0] = (color0->channels.r & 0xf8) | two_compl_trans_table[delta_r + 4];
	block[1] = (color0->channels.g & 0xf8) | two_compl_trans_table[delta_g + 4];
	block[2] = (color0->channels.b & 0xf8) | two_compl_trans_table[delta_b + 4];
}

// Writes the codeword table index for a sub-block into the control bits.
inline void WriteCodewordTable(uchar* block,
							   uchar sub_block_id,
							   uchar table) {
	uchar shift = (2 + (3 - sub_block_id * 3));
	block[3] &= ~(0x07 << shift);
	block[3] |= table << shift;
}

// Writes the 32-bit packed pixel indices to the compressed block.
inline void WritePixelData(uchar* block, int pixel_data) {
	block[4] |= pixel_data >> 24;
	block[5] |= (pixel_data >> 16) & 0xff;
	block[6] |= (pixel_data >> 8) & 0xff;
	block[7] |= pixel_data & 0xff;
}

// Writes the 'flip' bit, which determines sub-block partitioning (2x4 vs 4x2).
inline void WriteFlip(uchar* block, uchar flip) {
	block[3] &= ~0x01;
	block[3] |= flip;
}

// Writes the 'diff' bit, determining if differential color encoding is used.
inline void WriteDiff(uchar* block, uchar diff) {
	block[3] &= ~0x02;
	block[3] |= (diff) << 1;
}

// Simple memcpy implementation for device code.
void my_memcpy(void *dest, void *src, uint n) {
   char *csrc = (char *)src;
   char *cdest = (char *)dest;
   for (int i=0; i<n; i++)
       cdest[i] = csrc[i];
}

// Extracts a 4x4 block of pixels from the source image.
inline void ExtractBlock(uchar* dst, uchar* src, int width) {
	for (int j = 0; j < 4; ++j) {
		my_memcpy(&dst[j * 4 * 4], (void *)src, 4 * 4);
		src += width * 4;
	}
}

// Creates a 4-bit-per-channel color, expanding it to 8-bit.
inline union Color makeColor444(float* bgr) {
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

// Creates a 5-bit-per-channel color, expanding it to 8-bit.
inline union Color makeColor555(float* bgr) {
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

// Computes the average color of an 8-pixel sub-block.
void getAverageColor(union Color* src, float* avg_color) {
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
	
/**
 * @brief Finds the best luminance table and modifiers for a sub-block.
 * @return The total error for the best configuration.
 *
 * This function is the core of the compression logic. It performs a search over all
 * codeword tables and modifiers to find the combination that minimizes color error
 * for the 8 pixels in the sub-block (a form of Vector Quantization).
 */
unsigned long computeLuminance(uchar* block,
						    union Color* src,
						    union Color* base,
						   int sub_block_id,
						   constant uchar* idx_to_num_tab,
						   unsigned long threshold)
{
	uint best_tbl_err = threshold;
	uchar best_tbl_idx = 0;
	uchar best_mod_idx[8][8];  

	// Block Logic: Iterate through all 8 codeword tables to find the best match.
	for (uint tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		// Pre-calculate the 4 candidate colors for the current table.
		union Color candidate_color[4];  
		for (uint mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			candidate_color[mod_idx] = makeColor(base, lum);
		}
		
		uint tbl_err = 0;
		// For each pixel in the sub-block, find the best modifier.
		for (uint i = 0; i < 8; ++i) {
			uint best_mod_err = threshold;
			for (uint mod_idx = 0; mod_idx < 4; ++mod_idx) {
				 union Color *color = &candidate_color[mod_idx];
				uint mod_err = getColorError(&src[i], color);
				if (mod_err < best_mod_err) {
					best_mod_idx[tbl_idx][i] = mod_idx;
					best_mod_err = mod_err;
					if (mod_err == 0) break;  // Early exit if a perfect match is found.
				}
			}
			tbl_err += best_mod_err;
			if (tbl_err > best_tbl_err) break;  // Early exit if error exceeds the current best.
		}
		
		// If the current table gives a lower total error, update the best choice.
		if (tbl_err < best_tbl_err) {
			best_tbl_err = tbl_err;
			best_tbl_idx = tbl_idx;
			if (tbl_err == 0) break; // Early exit for a perfect table match.
		}
	}

	// Write the chosen table index and pack the pixel data.
	WriteCodewordTable(block, sub_block_id, best_tbl_idx);
	uint pix_data = 0;
	for (uint i = 0; i < 8; ++i) {
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

void* my_memset(void* pointer, int c, int size) {
    if ( pointer != NULL && size > 0 ) {
        uchar* pChar = (uchar*)pointer;
        for (int i = 0; i < size; ++i) {
            *pChar++ = (unsigned char) c; 
        }
    }
    return pointer;
}

// Attempts to compress a 4x4 block as a single solid color.
uchar tryCompressSolidBlock(uchar* dst,
						    union Color* src,
						   unsigned long* error)
{
	uchar my_true = 0x01;
	uchar my_false = 0x00;
	// Pre-condition: Check if all 16 pixels are identical.
	for (uint i = 1; i < 16; ++i) {
		if (src[i].bits != src[0].bits)
			return my_false;
	}
	
	my_memset(dst, 0, 8);
	float src_color_float[3] = {(float)(src->channels.b), (float)(src->channels.g), (float)(src->channels.r)};
	union Color base = makeColor555(src_color_float);
	
	WriteDiff(dst, my_true);
	WriteFlip(dst, my_false);
	WriteColors555(dst, &base, &base);
	
	uchar best_tbl_idx = 0;
	uchar best_mod_idx = 0;
	uint best_mod_err = UINT_MAX; 
	
	// Find the best luminance modifier for the solid color.
	for (uint tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		for (uint mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			union Color color = makeColor(&base, lum);
			uint mod_err = getColorError(src, &color);
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
	for (uint i = 0; i < 2; ++i) {
		for (uint j = 0; j < 8; ++j) {
			int texel_num = g_idx_to_num[i][j];
			pix_data |= msb << (texel_num + 16);
			pix_data |= lsb << (texel_num);
		}
	}
	WritePixelData(dst, pix_data);
	*error = 16 * best_mod_err;
	return my_true;
}

// Main compression function for a 4x4 block.
unsigned long compressBlock(uchar* dst,  union Color* ver_src,
									union Color* hor_src,
									unsigned long threshold)
{
	unsigned long solid_error = 0;
	// Fast path for solid color blocks.
	if (tryCompressSolidBlock(dst, ver_src, &solid_error)) {
		return solid_error;
	}
	
	union Color* sub_block_src[4] = {ver_src, ver_src + 8, hor_src, hor_src + 8};
	union Color sub_block_avg[4];
	uchar use_differential[2] = {0x01, 0x01};
	
	// Determine if differential mode should be used for each pair of sub-blocks.
	for (uint i = 0, j = 1; i < 4; i += 2, j += 2) {
		float avg_color_0[3];
		getAverageColor(sub_block_src[i], avg_color_0);
		union Color avg_color_555_0 = makeColor555(avg_color_0);
		
		float avg_color_1[3];
		getAverageColor(sub_block_src[j], avg_color_1);
		union Color avg_color_555_1 = makeColor555(avg_color_1);

		for (uint light_idx = 0; light_idx < 3; ++light_idx) {
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
	
	// Determine the best flip mode (partitioning) by comparing errors.
	uint sub_block_err[4] = {0};
	for (uint i = 0; i < 4; ++i) {
		for (uint j = 0; j < 8; ++j) {
			sub_block_err[i] += getColorError(&sub_block_avg[i], &sub_block_src[i][j]);
		}
	}
	
	uchar flip = (sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1]) ? 0x01 : 0x00;
	
	my_memset(dst, 0, 8);
	WriteDiff(dst, use_differential[!!flip]);
	WriteFlip(dst, flip);

	uchar sub_block_off_0 = (flip == 0x01) ? 2 : 0;
	uchar sub_block_off_1 = sub_block_off_0 + 1;
	
	if (use_differential[!!flip]) {
		WriteColors555(dst, &sub_block_avg[sub_block_off_0], &sub_block_avg[sub_block_off_1]);
	} else {
		WriteColors444(dst, &sub_block_avg[sub_block_off_0], &sub_block_avg[sub_block_off_1]);
	}
	
	// Compress the two chosen sub-blocks.
	unsigned long lumi_error1 = computeLuminance(dst, sub_block_src[sub_block_off_0], &sub_block_avg[sub_block_off_0], 0, g_idx_to_num[sub_block_off_0], threshold);
	unsigned long lumi_error2 = computeLuminance(dst, sub_block_src[sub_block_off_1], &sub_block_avg[sub_block_off_1], 1, g_idx_to_num[sub_block_off_1], threshold);
	
	return lumi_error1 + lumi_error2;
}

/**
 * @brief The main OpenCL kernel for texture compression.
 *
 * @param matSRC Input image data in global memory.
 * @param matDST Output buffer for compressed block data in global memory.
 * @param height Height of the source image.
 * @param width Width of the source image.
 *
 * Each work-item in the 2D NDRange is responsible for compressing a single
 * 4x4 block of the source image.
 */
__kernel void
compress(__global uchar* matSRC,
		__global uchar* matDST,
		int height,
		int width)
{
	union Color ver_blocks[16];
	union Color hor_blocks[16];

	// Threading Model: Use a 2D grid where each work-item corresponds to a 4x4 block.
	int y = get_global_id(0); // y-coordinate of the block.
	int x = get_global_id(1); // x-coordinate of the block.

	// Data Gathering: Manually copy pixel data from global to private memory.
	// This constructs the 'ver_blocks' and 'hor_blocks' needed for flip mode evaluation.
	int size = 8;
	uchar* cpy_dst;
	for (int i = 0; i < 4; i++) {
		cpy_dst = (uchar *)(ver_blocks + i * 2);
		for (int j = 0; j < size; j++) {
			cpy_dst[j] = matSRC[y * width * 4 * 4 + width * 4 * i + x * 4 * 4 + j];
		}
	}
	for (int i = 0; i < 4; i++) {
		cpy_dst = (uchar *)(ver_blocks + (8 + i * 2));
		for (int j = 0; j < size; j++) {
			cpy_dst[j] = matSRC[y * width * 4 * 4 + width * 4 * i + 2 * 4 + x * 4 * 4 + j];
		}
	}
	size = 16;
	for (int i = 0; i < 4; i++) {
		cpy_dst = (uchar *)(hor_blocks + i * 4);
		for (int j = 0; j < size; j++) {
			cpy_dst[j] = matSRC[y * width * 4 * 4 + width * 4 * i + x * 4 * 4 + j];
		}
	}

	// Perform the compression on the local block data.
	uchar dst[8];
	compressBlock(dst, ver_blocks, hor_blocks, INT_MAX);

	// Write the 8-byte compressed block to the correct location in global memory.
	for (int i = 0; i < 8; i++)
		matDST[y * width * 2 + 8 *x + i] = dst[i];
}

// --- C++ Host Code ---

#include "compress.hpp"

using namespace std;

#define DIE(assertion, call_description)                    
do {                                                        
    if (assertion) {                                        
            fprintf(stderr, "(%d): ",                       
                            __LINE__);                      
            perror(call_description);                       
            exit(EXIT_FAILURE);                             
    }                                                       
} while(0);

// Utility to convert OpenCL error codes to human-readable strings.
const char* cl_get_string_err(cl_int err) {
switch (err) {
  case CL_SUCCESS:                     	return  "Success!";
  // ... (many error cases)
  default:                                return  "Unknown";
  }
}

// Utility to get and print the OpenCL compiler build log on failure.
void cl_get_compiler_err_log(cl_program program, cl_device_id device)
{
	char* build_log;
	size_t log_size;
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
	build_log = new char[ log_size + 1 ];
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
	build_log[ log_size ] = '\0';
	cout << endl << build_log << endl;
}

// Error checking wrapper for OpenCL compilation.
int CL_COMPILE_ERR(int cl_ret, cl_program program, cl_device_id device)
{
	if(cl_ret != CL_SUCCESS){
		cout << endl << cl_get_string_err(cl_ret) << endl;
		cl_get_compiler_err_log(program, device);
		return 1;
	}
	return 0;
}

// General error checking wrapper for OpenCL calls.
static int CL_ERR(int cl_ret)
{
	if(cl_ret != CL_SUCCESS){
		cout << endl << cl_get_string_err(cl_ret) << endl;
		return 1;
	}
	return 0;
}

/**
 * @brief Constructor for TextureCompressor.
 * Architectural Intent: Initializes the OpenCL environment by discovering
 * platforms and selecting a GPU device to run the computations on.
 */
TextureCompressor::TextureCompressor() {
	cl_platform_id platform;
	cl_uint platform_num = 0;
	cl_platform_id* platform_list = NULL;
	cl_uint device_num = 0;
	cl_device_id* device_list = NULL;
	
	CL_ERR( clGetPlatformIDs(0, NULL, &platform_num));
	platform_list = new cl_platform_id[platform_num];
	DIE(platform_list == NULL, "alloc platform_list");
	CL_ERR( clGetPlatformIDs(platform_num, platform_list, NULL));
	
	for(uint platf=0; platf<platform_num; platf++) {
		platform = platform_list[platf];
		DIE(platform == 0, "platform selection");
		
		if(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &device_num) != CL_SUCCESS) {
			device_num = 0;
			continue;
		}
		device_list = new cl_device_id[device_num];
		DIE(device_list == NULL, "alloc devices");
		CL_ERR( clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, device_num, device_list, NULL));
	
		for(uint dev=0; dev<device_num; dev++) {
			cl_device_type devType;
			clGetDeviceInfo(device_list[dev], CL_DEVICE_TYPE, sizeof(cl_device_type), &devType, NULL);
			
			if (devType == CL_DEVICE_TYPE_GPU){
				this->device = device_list[dev];
				this->device_ids = device_list;
			}
		}
	}
	this->device_ids = device_list;
	this->platform_ids = platform_list;
} 	

TextureCompressor::~TextureCompressor() {
	delete[] platform_ids;
	delete[] device_ids;
}	

// Helper to read kernel source from a file.
static void read_kernel(string file_name, string &str_kernel)
{
	ifstream in_file(file_name.c_str());
	in_file.open(file_name.c_str());
	DIE( !in_file.is_open(), "ERR OpenCL kernel file. Same directory as binary ?" );

	stringstream str_stream;
	str_stream << in_file.rdbuf();
	str_kernel = str_stream.str();
}

/**
 * @brief Compresses an image using the OpenCL kernel.
 *
 * This method orchestrates the entire compression process, from setting up
 * OpenCL objects to executing the kernel and retrieving the results.
 */
unsigned long TextureCompressor::compress(const uint8_t* src,
									  uint8_t* dst,
									  int width,
									  int height)
{
	size_t	global[2];
	string	kernel_src;
	cl_context context;
	cl_command_queue commands;
	cl_program program;
	cl_kernel kernel;
	int ret;
	unsigned int size_src = width * height * 4;
	unsigned int size_dst = width * height * 4 / 8;

	// Step 1: Create OpenCL context and command queue.
	context = clCreateContext(0, 1, &this->device, NULL, NULL, &ret);
	CL_ERR( ret );
	commands = clCreateCommandQueue(context, this->device, CL_QUEUE_PROFILING_ENABLE, &ret);
	CL_ERR( ret );

	// Step 2: Create memory buffers on the device for source and destination.
	cl_mem memSRC, memDST;
	memSRC = clCreateBuffer(context,  CL_MEM_READ_ONLY, sizeof(uint8_t) * size_src, NULL, NULL);
	memDST = clCreateBuffer(context,  CL_MEM_WRITE_ONLY, sizeof(uint8_t) * size_dst, NULL, NULL);

	// Step 3: Copy source data from host to device.
	ret = clEnqueueWriteBuffer(commands, memSRC, CL_TRUE, 0, sizeof(uint8_t) * size_src, src, 0, NULL, NULL);

	// Step 4: Read, create, and build the OpenCL program.
	read_kernel("sol_device.cl", kernel_src);
	const char* kernel_c_str = kernel_src.c_str();
	program = clCreateProgramWithSource(context, 1, (const char **) &kernel_c_str, NULL, &ret);
	CL_ERR( ret );
	ret = clBuildProgram(program, 1, &this->device, NULL, NULL, NULL);
	CL_COMPILE_ERR( ret, program, this->device );
	kernel = clCreateKernel(program, "compress", &ret);
	CL_ERR( ret );

	// Step 5: Set the arguments for the kernel.
	ret  = 0;
	ret |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &memSRC);
	ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &memDST);
	ret |= clSetKernelArg(kernel, 2, sizeof(int), &height);
	ret |= clSetKernelArg(kernel, 3, sizeof(int), &width);

	// Step 6: Define the NDRange (work-items grid) and execute the kernel.
	cl_event prof_event;
	global[0] = (size_t)height / 4;
	global[1] = (size_t)width / 4;
	ret = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, global, NULL, 0, NULL, &prof_event);	

	// Step 7: Read the compressed data back from device to host.
	ret = clEnqueueReadBuffer( commands, memDST, CL_TRUE, 0, sizeof(uint8_t) * size_dst, dst, 0, NULL, NULL );

	// Step 8: Wait for all commands to finish and clean up resources.
	clFinish(commands);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseMemObject(memSRC);
	clReleaseMemObject(memDST);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);

    // Note: This function doesn't return the error metric like the previous file.
    // It should, but we are not modifying the code.
    return 0;
}
