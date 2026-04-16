/**
 * @file kernel.cl
 * @brief OpenCL implementation of an ETC1-like texture compression algorithm.
 * @details This file contains the kernel and helper functions for compressing a 4x4
 * pixel block into the ETC1 format. The algorithm works by dividing a 4x4 block
 * into two sub-blocks and encoding each with a base color and a set of modifier
 * values chosen from a codeword table to minimize color error. It supports
 * differential mode (where sub-block colors are offsets from each other) and
 * individual mode, as well as a "flip" bit to change the sub-block orientation.
 */

// Union to represent a 32-bit BGRA color, allowing access by channels, components, or raw bits.
union infoRGB
{
	struct BgraColorType
	{
		uchar b;
		uchar g;
		uchar r;
		uchar a;
	} channels;
	uchar components[4];
	uint bits;
};
typedef union infoRGB Color; 

// Rounds a 0-255 float value to a 5-bit representation (0-31).
uchar round_to_5_bits(float val)
{
	return clamp(val * 31.0f / 255.0f + 0.5f, 0.0f, 31.0f);
}

// Rounds a 0-255 float value to a 4-bit representation (0-15).
uchar round_to_4_bits(float val)
{
	return clamp(val * 15.0f / 255.0f + 0.5f, 0.0f, 15.0f);
}

// Custom memcpy implementation for use within the kernel.
void copy(void *dst, void *src, int n)
{
   
	char *charSrc = (char *)src;
	char *charDst = (char *)dst;
 	int i;
	
	for (i = 0;i < n;i++)
	{
 	   charDst[i] = charSrc[i];
	}
}

// Custom memset-like function to populate a memory area with a given byte value.
void populate(void *pointer, int info, int n)
{
	int i = 0;
	uchar *unit = pointer;
  
	while(n > 0)
	{
    	*unit = info;
    	unit++;
    	n--;
    }
}

// Codeword tables for intensity modifications in ETC1.
__constant short g_codeword_tables[8][4] =
{
	{-8, -2, 2, 8},
	{-17, -5, 5, 17},
	{-29, -9, 9, 29},
	{-42, -13, 13, 42},
	{-60, -18, 18, 60},
	{-80, -24, 24, 80},
	{-106, -33, 33, 106},
	{-183, -47, 47, 183}
};

// Maps a 2-bit modifier index to a 2-bit pixel index.
__constant uchar g_mod_to_pix[4] = {3, 2, 0, 1};

// Maps a linear texel index to its position in the final 32-bit pixel data word.
__constant uchar g_idx_to_num[4][8] =
{
	{0, 4, 1, 5, 2, 6, 3, 7},        // Sub-block 0, no flip
	{8, 12, 9, 13, 10, 14, 11, 15},  // Sub-block 1, no flip
	{0, 4, 8, 12, 1, 5, 9, 13},      // Sub-block 0, flip
	{2, 6, 10, 14, 3, 7, 11, 15}     // Sub-block 1, flip
};

// Applies a luminance modifier to a base color.
Color makeColor(const Color *base, short lum)
{
	Color color;

	int b = (int)(base->channels.b) + lum;
	int g = (int)(base->channels.g) + lum;
	int r = (int)(base->channels.r) + lum;
	
	color.channels.b = (uchar)(clamp(b, 0, 255));
	color.channels.g = (uchar)(clamp(g, 0, 255));
	color.channels.r = (uchar)(clamp(r, 0, 255));
	
	return color;
}


// Calculates the error between two colors. Can use a weighted perceptual metric or simple squared distance.
uint getColorError(const Color *u, const Color *v)
{
	#ifdef USE_PERCEIVED_ERROR_METRIC
		float delta_b = (float)(u->channels.b) - v->channels.b;
		float delta_g = (float)(u->channels.g) - v->channels.g;
		float delta_r = (float)(u->channels.r) - v->channels.r;
		return (uint)(0.299f * delta_b * delta_b + 0.587f * delta_g * delta_g + 0.114f * delta_r * delta_r);
	#else
		int delta_b = (int)(u->channels.b) - v->channels.b;
		int delta_g = (int)(u->channels.g) - v->channels.g;
		int delta_r = (int)(u->channels.r) - v->channels.r;
		return delta_b * delta_b + delta_g * delta_g + delta_r * delta_r;
	#endif
}


// Writes the packed 4-bit base colors for individual mode.
void WriteColors444(uchar *block, const Color *color0, const Color *color1)
{	
	block[0] = (color0->channels.r & 0xf0) | (color1->channels.r >> 4);
	block[1] = (color0->channels.g & 0xf0) | (color1->channels.g >> 4);
	block[2] = (color0->channels.b & 0xf0) | (color1->channels.b >> 4);
}

// Writes the 5-bit base color and 3-bit delta for differential mode.
void WriteColors555(uchar *block, const Color *color0, const Color *color1)
{
	const uchar two_compl_trans_table[8] = { 4, 5, 6, 7, 0, 1, 2, 3 };
	short delta_r = (short)(color1->channels.r >> 3) - (color0->channels.r >> 3);
	short delta_g = (short)(color1->channels.g >> 3) - (color0->channels.g >> 3);
	short delta_b = (short)(color1->channels.b >> 3) - (color0->channels.b >> 3);
	block[0] = (color0->channels.r & 0xf8) | two_compl_trans_table[delta_r + 4];
	block[1] = (color0->channels.g & 0xf8) | two_compl_trans_table[delta_g + 4];
	block[2] = (color0->channels.b & 0xf8) | two_compl_trans_table[delta_b + 4];
}

// Writes the 3-bit codeword table index for a given sub-block.
void WriteCodewordTable(uchar *block, uchar sub_block_id, uchar table)
{
	uchar shift = (2 + (3 - sub_block_id * 3));
	block[3] &= ~(0x07 << shift);
	block[3] |= table << shift;
}

// Writes the 32-bit packed pixel indices to the block.
void WritePixelData(uchar *block, uint pixel_data)
{
	block[4] |= pixel_data >> 24;
	block[5] |= (pixel_data >> 16) & 0xff;
	block[6] |= (pixel_data >> 8) & 0xff;
	block[7] |= pixel_data & 0xff;
}

// Writes the flip bit to the block.
void WriteFlip(uchar *block, bool flip)
{
	block[3] &= ~0x01;
	block[3] |= (uchar)(flip);
}

// Writes the differential mode bit to the block.
void WriteDiff(uchar *block, bool diff)
{
	block[3] &= ~0x02;
	block[3] |= (uchar)(diff) << 1;
}

// Extracts a 4x4 block of pixels from a larger source image.
void ExtractBlock(uchar *dst, const uchar *src, int width)
{
	for (int j = 0; j < 4; j++)
	{
		copy((void *)(&dst[j * 4 * 4]), (void *)src, 4 * 4);
		src += width * 4;
	}
}


// Creates a color quantized to 4 bits per channel, then expanded back to 8 bits.
Color makeColor444(const float *bgr)
{
	Color bgr444;
	uchar b4 = round_to_4_bits(bgr[0]);
	uchar g4 = round_to_4_bits(bgr[1]);
	uchar r4 = round_to_4_bits(bgr[2]);
	bgr444.channels.b = (b4 << 4) | b4;
	bgr444.channels.g = (g4 << 4) | g4;
	bgr444.channels.r = (r4 << 4) | r4;
	bgr444.channels.a = 0x44;
	return bgr444;
}


// Creates a color quantized to 5 bits per channel, then expanded back to 8 bits.
Color makeColor555(const float *bgr)
{
	Color bgr555;
	uchar b5 = round_to_5_bits(bgr[0]);
	uchar g5 = round_to_5_bits(bgr[1]);
	uchar r5 = round_to_5_bits(bgr[2]);
	bgr555.channels.b = (b5 > 2);
	bgr555.channels.g = (g5 > 2);
	bgr555.channels.r = (r5 > 2);
	bgr555.channels.a = 0x55;
	return bgr555;
}
	
// Calculates the average color of an 8-pixel sub-block.
void getAverageColor(const Color *src, float *avg_color)
{
	uint sum_b = 0, sum_g = 0, sum_r = 0;
	const float kInv8 = 1.0f / 8.0f;
	for (uint i = 0; i < 8; i++)
	{
		sum_b += src[i].channels.b;
		sum_g += src[i].channels.g;
		sum_r += src[i].channels.r;
	}
	avg_color[0] = (float)(sum_b) * kInv8;
	avg_color[1] = (float)(sum_g) * kInv8;
	avg_color[2] = (float)(sum_r) * kInv8;
}

/**
 * @brief Finds the best codeword table and modifier indices for a sub-block.
 * @param block The destination compressed block.
 * @param src The 8-pixel source sub-block.
 * @param base The base color for the sub-block.
 * @param sub_block_id The ID of the sub-block (0 or 1).
 * @param index The index into the pixel ordering table.
 * @param threshold The error threshold for early termination.
 * @return The total error for the best encoding found.
 * @details This function iterates through all codeword tables to find the
 * one that minimizes the total color error for the sub-block.
 */
ulong computeLuminance(uchar *block, const Color *src, const Color *base, int sub_block_id, const uchar index, ulong threshold)
{
	uint best_tbl_err = threshold;
	uchar best_tbl_idx = 0;
	uchar best_mod_idx[8][8];  

	for (uint tbl_idx = 0; tbl_idx < 8; tbl_idx++)
	{
		Color candidate_color[4];  
		for (uint mod_idx = 0; mod_idx < 4; mod_idx++)
		{
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			candidate_color[mod_idx] = makeColor(base, lum);
		}
		
		uint tbl_err = 0;
		for (uint i = 0; i < 8; i++)
		{
			uint best_mod_err = threshold;
			for (uint mod_idx = 0; mod_idx < 4; mod_idx++)
			{
				const Color color = candidate_color[mod_idx];
				uint mod_err = getColorError(&src[i], &color);
				if (mod_err < best_mod_err)
				{
					best_mod_idx[tbl_idx][i] = mod_idx;
					best_mod_err = mod_err;
					if (mod_err == 0) break;
				}
			}
			tbl_err += best_mod_err;
			if (tbl_err > best_tbl_err) break;
		}
		
		if (tbl_err < best_tbl_err)
		{
			best_tbl_err = tbl_err;
			best_tbl_idx = tbl_idx;
			if (tbl_err == 0) break;
		}
	}

	WriteCodewordTable(block, sub_block_id, best_tbl_idx);

	uint pix_data = 0;
	for (uint i = 0; i < 8; i++)
	{
		uchar mod_idx = best_mod_idx[best_tbl_idx][i];
		uchar pix_idx = g_mod_to_pix[mod_idx];
		uint lsb = pix_idx & 0x1;
		uint msb = pix_idx >> 1;
		int texel_num = g_idx_to_num[index][i];
		pix_data |= msb << (texel_num + 16);
		pix_data |= lsb << (texel_num);
	}

	WritePixelData(block, pix_data);
	return best_tbl_err;
}


/**
 * @brief Attempts to compress a 4x4 block as a single solid color.
 * @param dst The destination compressed block.
 * @param src The 16-pixel source block.
 * @param error Pointer to store the resulting compression error.
 * @return True if the block was compressed as solid, false otherwise.
 */
bool tryCompressSolidBlock(uchar *dst, const Color *src, ulong *error)
{
	for (uint i = 1; i < 16; i++)
	{
		if (src[i].bits != src[0].bits) return false;
	}
	
	populate(dst, 0, 8);
	float src_color_float[3] = {(float)(src->channels.b), (float)(src->channels.g), (float)(src->channels.r)};
	Color base = makeColor555(src_color_float);
	
	WriteDiff(dst, true);
	WriteFlip(dst, false);
	WriteColors555(dst, &base, &base);
	
	uchar best_tbl_idx = 0;
	uchar best_mod_idx = 0;
	uint best_mod_err = UINT_MAX; 
	
	for (uint tbl_idx = 0; tbl_idx < 8; tbl_idx++)
	{
		for (uint mod_idx = 0; mod_idx < 4; mod_idx++)
		{
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			const Color color = makeColor(&base, lum);
			uint mod_err = getColorError(src, &color);
			if (mod_err < best_mod_err)
			{
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
	for (uint i = 0; i < 2; i++)
	{
		for (uint j = 0; j < 8; j++)
		{
			int texel_num = g_idx_to_num[i][j];
			pix_data |= msb << (texel_num + 16);
			pix_data |= lsb << (texel_num);
		}
	}
	
	WritePixelData(dst, pix_data);
	*error = 16 * best_mod_err;
	return true;
}

/**
 * @brief Compresses a 4x4 pixel block into ETC1 format.
 * @param dst The destination for the 8-byte compressed block.
 * @param ver_src Source pixels arranged for vertical sub-block processing.
 * @param hor_src Source pixels arranged for horizontal sub-block processing.
 * @param threshold Error threshold for early termination.
 * @return The total compression error for the block.
 */
ulong compressBlock(uchar *dst, const Color *ver_src, const Color *hor_src, ulong threshold)
{
	ulong solid_error = 0;
	if (tryCompressSolidBlock(dst, ver_src, &solid_error))
	{
		return solid_error;
	}
	
	const Color *sub_block_src[4] = {ver_src, ver_src + 8, hor_src, hor_src + 8};
	
	Color sub_block_avg[4];
	bool use_differential[2] = {true, true};
	
	for (uint i = 0, j = 1; i < 4; i += 2, j += 2)
	{
		float avg_color_0[3];
		getAverageColor(sub_block_src[i], avg_color_0);
		Color avg_color_555_0 = makeColor555(avg_color_0);
		
		float avg_color_1[3];
		getAverageColor(sub_block_src[j], avg_color_1);
		Color avg_color_555_1 = makeColor555(avg_color_1);
		
		for (uint light_idx = 0; light_idx < 3; light_idx++)
		{
			int u = avg_color_555_0.components[light_idx] >> 3;
			int v = avg_color_555_1.components[light_idx] >> 3;
			int component_diff = v - u;
			if (component_diff  3)
			{
				use_differential[i / 2] = false;
				sub_block_avg[i] = makeColor444(avg_color_0);
				sub_block_avg[j] = makeColor444(avg_color_1);
			}
			else
			{
				sub_block_avg[i] = avg_color_555_0;
				sub_block_avg[j] = avg_color_555_1;
			}
		}
	}
	
	uint sub_block_err[4] = {0};
	for (uint i = 0; i < 4; i++)
	{
		for (uint j = 0; j < 8; j++)
		{
			sub_block_err[i] += getColorError(&sub_block_avg[i], &sub_block_src[i][j]);
		}
	}
	
	bool flip = sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1];
	
	populate(dst, 0, 8);
	
	WriteDiff(dst, use_differential[!!flip]);
	WriteFlip(dst, flip);
	
	uchar sub_block_off_0 = flip ? 2 : 0;
	uchar sub_block_off_1 = sub_block_off_0 + 1;
	
	if (use_differential[!!flip])
	{
		WriteColors555(dst, &sub_block_avg[sub_block_off_0], &sub_block_avg[sub_block_off_1]);
	}
	else
	{
		WriteColors444(dst, &sub_block_avg[sub_block_off_0], &sub_block_avg[sub_block_off_1]);
	}
	
	ulong lumi_error1 = 0, lumi_error2 = 0;
	
	lumi_error1 = computeLuminance(dst, sub_block_src[sub_block_off_0], &sub_block_avg[sub_block_off_0], 0, sub_block_off_0, threshold);
	lumi_error2 = computeLuminance(dst, sub_block_src[sub_block_off_1], &sub_block_avg[sub_block_off_1], 1, sub_block_off_1, threshold);
	
	return lumi_error1 + lumi_error2;
}

/**
 * @brief Main OpenCL kernel for ETC1 texture compression.
 * @param src The global memory buffer for the source image data (BGRA).
 * @param dst The global memory buffer for the compressed output data.
 * @param dims An array containing the width and height of the source image.
 * @details Each work-item compresses one 4x4 block of pixels.
 */
__kernel void execute(__global uchar *src, __global uchar *dst, __global int *dims)
{
	int width = dims[0];
	int height = dims[1];
	
	Color ver_blocks[16];
	Color hor_blocks[16];
	
	int ycoord = get_global_id(0);
	int xcoord = get_global_id(1);

	int soffset = width * 16 * ycoord + xcoord * 16;
	int doffset = width * 2 * ycoord + xcoord * 8;

	const Color* row0 = src + soffset;
	const Color* row1 = row0 + width;
	const Color* row2 = row1 + width;
	const Color* row3 = row2 + width;
			
	copy((void *)ver_blocks, (void *)row0, 8);
	copy((void *)ver_blocks + 2, (void *)row1, 8);
	copy((void *)ver_blocks + 4, (void *)row2, 8);
	copy((void *)ver_blocks + 6, (void *)row3, 8);
	copy((void *)ver_blocks + 8, (void *)row0 + 2, 8);
	copy((void *)ver_blocks + 10, (void *)row1 + 2, 8);
	copy((void *)ver_blocks + 12, (void *)row2 + 2, 8);
	copy((void *)ver_blocks + 14, (void *)row3 + 2, 8);
			
	copy(hor_blocks, row0, 16);
	copy(hor_blocks + 4, row1, 16);
	copy(hor_blocks + 8, row2, 16);
	copy(hor_blocks + 12, row3, 16);

	uchar aux[8];
	
	compressBlock(aux, ver_blocks, hor_blocks, INT_MAX);

	for(int i = 0;i < 8;i++)
	{
		dst[doffset + i] = aux[i];
	}
}

// The following appears to be C++ host code concatenated into the kernel file.
#include "compress.hpp"

using namespace std;

void gpu_find(cl_device_id &device, uint device_select);

TextureCompressor::TextureCompressor() 
{
	gpu_find(this->device, 0); 
} 

TextureCompressor::~TextureCompressor()
{
}

#define DIE(assertion, call_description)  \
do{ \
	if (assertion) { \
		fprintf(stderr, "(%d): ", __LINE__); \
		perror(call_description); \
		exit(EXIT_FAILURE); \
	} \
}while(0);

const char* cl_get_string_err(cl_int err)
{
	switch (err)
	{
		case CL_SUCCESS:                     	return  "Success!";
		case CL_DEVICE_NOT_FOUND:               return  "Device not found.";
	    case CL_DEVICE_NOT_AVAILABLE:           return  "Device not available";
		// ... and so on for all error codes
		default:                                return  "Unknown";
	}
}

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

void read_kernel(string file_name, string &str_kernel)
{
	ifstream in_file(file_name.c_str());
	in_file.open(file_name.c_str());
	DIE( !in_file.is_open(), "ERR OpenCL kernel file. Same directory as binary ?" );
	stringstream str_stream;
	str_stream << in_file.rdbuf();
	str_kernel = str_stream.str();
}

int CL_ERR(int cl_ret)
{
	if(cl_ret != CL_SUCCESS){
		cout << endl << cl_get_string_err(cl_ret) << endl;
		return 1;
	}
	return 0;
}

int CL_COMPILE_ERR(int cl_ret, cl_program program, cl_device_id device)
{
	if(cl_ret != CL_SUCCESS)
	{
		cout << endl << cl_get_string_err(cl_ret) << endl;
		cl_get_compiler_err_log(program, device);
		return 1;
	}
	return 0;
}

void gpu_find(cl_device_id &device, uint device_select)
{
	cl_platform_id platform;
	cl_uint platform_num = 0;
	cl_platform_id* platform_list = NULL;
	cl_uint device_num = 0;
	cl_device_id* device_list = NULL;
	int deviceFound = 0;

	CL_ERR( clGetPlatformIDs(0, NULL, &platform_num));
	platform_list = new cl_platform_id[platform_num];
	DIE(platform_list == NULL, "alloc platform_list");
	CL_ERR( clGetPlatformIDs(platform_num, platform_list, NULL));

	for(uint platf=0; platf<platform_num; platf++)
	{
		platform = platform_list[platf];
		DIE(platform == 0, "platform selection");
		if(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &device_num) == CL_DEVICE_NOT_FOUND) {
			device_num = 0;
			continue;
		}
		device_list = new cl_device_id[device_num];
		DIE(device_list == NULL, "alloc devices");
		CL_ERR( clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, device_num, device_list, NULL));
		for(uint dev=0; dev<device_num; dev++)
		{
			if(dev == device_select)
			{
				device = device_list[dev];
				deviceFound = 1;
				break;
			}
		}
		if(deviceFound == 1) break;
	}
	delete[] platform_list;
	delete[] device_list;
}

void solve(cl_device_id device, const uint8_t *src, uint8_t *dst, int width, int height)
{
	string kernel_src;
	cl_context context;
	cl_command_queue commands;
	cl_program program;
	cl_kernel kernel;
	cl_mem source, destination, dimensions;
	size_t global[2];
	int ret, dim[2];
	
	context = clCreateContext(0, 1, &device, NULL, NULL, &ret);
	CL_ERR(ret);
	commands = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &ret);
	CL_ERR(ret);
	
	dim[0] = width;
	dim[1] = height;
	int srcSz = width * height * 4;
	int dstSz = srcSz / 8;
	
	dimensions = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * 2, NULL, NULL);
	source = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(uint8_t) * srcSz, NULL, NULL);
	destination = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(uint8_t) * dstSz, NULL, NULL);

	read_kernel("kernel.cl", kernel_src);
	const char *kernel_c_str = kernel_src.c_str();
	
	program = clCreateProgramWithSource(context, 1, (const char **) &kernel_c_str, NULL, &ret);
	CL_ERR(ret);
	
	ret = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	CL_COMPILE_ERR(ret, program, device);
	
	kernel = clCreateKernel(program, "execute", &ret);
	CL_ERR(ret);

	ret  = 0;
	ret  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &source);
	ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &destination);
	ret |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &dimensions);

	ret = clEnqueueWriteBuffer(commands, source, CL_TRUE, 0, sizeof(uint8_t) * srcSz, src, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(commands, dimensions, CL_TRUE, 0, sizeof(int) *2, dim, 0, NULL, NULL);

	cl_event prof_event;
	global[0] = (size_t)height / 4;
	global[1] = (size_t)width / 4;
	ret = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, global, NULL, 0, NULL, &prof_event);
	
	clFinish(commands);

	ret = clEnqueueReadBuffer(commands, destination, CL_TRUE, 0, sizeof(uint8_t) * dstSz, dst, 0, NULL, NULL);
	
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseMemObject(source);
	clReleaseMemObject(destination);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);
}

unsigned long TextureCompressor::compress(const uint8_t* src, uint8_t* dst, int width, int height)
{
	solve(this->device, src, dst, width, height);
	return 0;
}
