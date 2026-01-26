
>>>> file: compare_ppm.cpp
#include 
#include 
#include 
#include 
#include 
#include 

using namespace std;

int main(int argc, char*args[]) {

	int width, height, max;
	char buffer[256];
	FILE *f[argc - 1];
	
	for (int i = 0; i< argc - 1; i++) {
		char *file = args[1 + i]; 
		f[i] = fopen(file, "rb");
		fgets(buffer, sizeof(buffer), f[i]);
		long data_start = ftell(f[i]);

		while (fgets(buffer, sizeof(buffer), f[i])) {
			if (buffer[0] == '#') {
				data_start = ftell(f[i]);
			}
			else {
				fseek(f[i], data_start, SEEK_SET);
				break;
			}
		}
		fscanf(f[i], "%d %d %d\n", &width, &height, &max);
	}

	uint8_t *src = (uint8_t*)calloc(width * height * 4, 1);
	uint8_t *src1 = (uint8_t*)calloc(width * height * 4, 1);
	unsigned long *error = (unsigned long*)calloc(argc - 2, sizeof(unsigned long));

	int stride = width * 4;
	int red, green, blue, alpha = 3;

	red = 2;
	green = 1;
	blue = 0;
	
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < stride; j += 4) {
			fread(&src[stride * i + j + red], 1, 1, f[0]);
			fread(&src[stride * i + j + green], 1, 1, f[0]);
			fread(&src[stride * i + j + blue], 1, 1, f[0]);
			src[stride * i + j + alpha] = 0;
		}
	}

	for (int k = 1; k < argc - 1; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < stride; j += 4) {
				fread(&src1[stride * i + j + red], 1, 1, f[k]);
				fread(&src1[stride * i + j + green], 1, 1, f[k]);
				fread(&src1[stride * i + j + blue], 1, 1, f[k]);
				src1[stride * i + j + alpha] = 0;
				for (int l = 0; l < 4; l++) {
					unsigned long temp = (src1[stride * i + j + l] - src[stride * i + j + l]);
					error[k - 1] += (unsigned long)(temp * temp);
				}
			}
		}
	}

	for (int i = 0; i < argc - 2; i++) {
		double PSNR = 10 * log10(3 * 255.0 * 255.0 / (error[i] / (double)(width * height)));
		double RMSE = sqrt(error[i] / (double)(width * height));
		printf("FILE = %s, \tPSNR = %.6lf, \tRMSE = %.6lf\n", args[i + 2], PSNR, RMSE);
	}
}
>>>> file: compress.cpp
#include "compress.hpp"

#include 
#include 
#include 
#include 
#include 
#include 
#include 

using namespace std;

#define BGRA

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


void read_kernel(std::string file_name, std::string &str_kernel)
{
	std::ifstream in_file(file_name.c_str());
	in_file.open(file_name.c_str());
	DIE( !in_file.is_open(), "ERR OpenCL kernel file. Same directory as binary ?" );

	std::stringstream str_stream;
	str_stream << in_file.rdbuf();

	str_kernel = str_stream.str();
}

int readCompressWrite(const char *input, 
	const char *output,
	TextureCompressor *compressor)
{
	char line[256];
	int red, green, blue;
	int max;
	int width, height;
	long data_start;
	int stride;
	int compressed_size;
	uint8_t *src, *dst;
	auto begin_timer = std::chrono::high_resolution_clock::now();

	FILE *f = fopen(input, "r");
	if (!f) {
		cout << "File not found: " << input << endl;
		return EXIT_FAILURE;
	}

#if defined(BGRA)
	red = 2;
	green = 1;
	blue = 0;
	#else
	red = 0;
	green = 1;
	blue = 2;
#endif

	fgets(line, sizeof(line), f);
	data_start = ftell(f);

	while (fgets(line, sizeof(line), f)) {
		if (line[0] == '#') {
			data_start = ftell(f);
		}
		else {
			fseek(f, data_start, SEEK_SET);
			break;
		}
	}

	fscanf(f, "%d %d %d\n", &width, &height, &max);
	src = (uint8_t*)malloc( width * height * 4);

	if (!src) {
		fclose(f);
		return EXIT_FAILURE;
	}

	stride = width * 4;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < stride; j += 4) {
			fread(&src[stride * i + j + red], 1, 1, f);
			fread(&src[stride * i + j + green], 1, 1, f);
			fread(&src[stride * i + j + blue], 1, 1, f);
			src[stride * i + j + 3] = 0;
		}
	}

	fclose(f);
	
	compressed_size = width * height * 4 / 8;
	dst = (uint8_t*)malloc( compressed_size);

	if (!dst) {
		free(src);
		return EXIT_FAILURE;
	}

	memset(dst, 0, compressed_size);
	
	// actual compression
	begin_timer = std::chrono::high_resolution_clock::now();
	compressor->compress(src, dst, width, height);

	// end timer compressor
	cout << "FILE = " << input << ", \tTIME.ms = " <<
	std::chrono::duration_cast
	(std::chrono::high_resolution_clock::now() - begin_timer).count()
	<< endl;

	// write file	
	f = fopen(output, "wb");
	if (f) {
		uint8_t header[16];
		memcpy(header, "PKM ", 4);
		memcpy(&header[4], "20", 2);
		header[6] = 0;
		header[7] = 1;
		header[8] = header[12] = (width >> 8) & 0xFF;
		header[9] = header[13] = width & 0xFF;
		header[10] = header[14] = (height >> 8) & 0xFF;
		header[11] = header[15] = height & 0xFF;

		fwrite(header, 16, 1, f);
		fwrite(dst, compressed_size, 1, f);
		fclose(f);
	}

	free(dst);
	free(src);
	return EXIT_SUCCESS;
}

struct CompressedFile
{
	string in;
	string out;
};

int main(int argc, const char **argv)
{
	int ret = 0;
	vector todoFiles;

	auto begin_timer = std::chrono::high_resolution_clock::now();
	
	// create compressor
	TextureCompressor *compressor = new TextureCompressor();
	if (!compressor) {
		cout << "ERR allocate compressor" << endl;
		return EXIT_FAILURE;
	}
	
	// end timer contructor
	cout << "INIT TIME.ms = " <<
	chrono::duration_cast
	(chrono::high_resolution_clock::now() - begin_timer).count()
	<< endl;
	
	if(compressor->device == 0) {
		cout << "DEVICE INVALID. Is compressor.device set ?" << endl;
	}
	else {
		char devName[512];
		clGetDeviceInfo(compressor->device, CL_DEVICE_NAME,
				512, devName, NULL);
		cout << "DEVICE = " << devName << endl;
		cl_device_type devType;
		clGetDeviceInfo(compressor->device, CL_DEVICE_TYPE,
            sizeof(cl_device_type), &devType, NULL);
		if(devType == CL_DEVICE_TYPE_GPU) {
			cout << "GPU = YES" << endl;
		} else {
			cout << "GPU = NO!!" << endl;	
		}
	}

	// cycle through list of files
	for(int i = 1; i < argc; i++) {
		CompressedFile cFile;
		cFile.in = argv[i];
		cFile.out = cFile.in + ".pkm";
		todoFiles.push_back(cFile);
	}
	for(auto cFile : todoFiles) {
		ret |= readCompressWrite(cFile.in.c_str(), 
			cFile.out.c_str(), compressor);
	}
	
	delete compressor;
	return ret;
}
>>>> file: compress.hpp
// Copyright 2015 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CC_RASTER_TEXTURE_COMPRESSOR_H_
#define CC_RASTER_TEXTURE_COMPRESSOR_H_

#include 
#include 
#include 
#include 
#include 
#include 
#if __APPLE__
#include 
#else
#include 
#endif

#define NOTREACHED()

#define DIE(assertion, call_description)                    \


do {                                                        \
    if (assertion) {                                        \
            fprintf(stderr, "(%d): ",                       \
                            __LINE__);                      \
            perror(call_description);                       \
            exit(EXIT_FAILURE);                             \
    }                                                       \
} while(0);

union Color {
	struct BgraColorType {
		uint8_t b;
		uint8_t g;
		uint8_t r;
		uint8_t a;
	} channels;
	uint8_t components[4];
	uint32_t bits;
};

// Defining the following macro will cause the error metric function to weigh
// each color channel differently depending on how the human eye can perceive
// them. This can give a slight improvement in image quality at the cost of a
// performance hit.
// #define USE_PERCEIVED_ERROR_METRIC

#define ALIGNAS(X)	__attribute__((aligned(X)))

void read_kernel(std::string file_name, std::string &str_kernel);
int CL_ERR(int cl_ret);
int CL_COMPILE_ERR(int cl_ret,
                  cl_program program,
                  cl_device_id device);

const char* cl_get_string_err(cl_int err);
void cl_get_compiler_err_log(cl_program program,
                             cl_device_id device);

class TextureCompressor {
	cl_context        context;
	cl_program        program;
	cl_command_queue  command_queue;
	cl_kernel         kernel;
	cl_device_id      *device_ids;
	cl_platform_id    *platform_ids;
	
public:
	cl_device_id 	device;	// used to inspect device type
	TextureCompressor();
	~TextureCompressor();
	
	// Compress a texture using ETC1. Note that the |quality| parameter is
	// ignored. The current implementation does not support different quality
	// settings.
	unsigned long compress(const uint8_t* src,
						   uint8_t* dst,
						   int width,
						   int height);

	void gpu_find(cl_device_id &device);
	
	
protected:
	unsigned long compressBlock(uint8_t* dst,
								const Color* ver_src,
								const Color* hor_src,
								unsigned long threshold);
};

#endif  // CC_RASTER_TEXTURE_COMPRESSOR_H_
>>>> file: sol_device.cl
typedef struct ColorType {
	uchar b;
	uchar g;
	uchar r;
	uchar a;
} ct;

typedef struct Color {
	struct ColorType channels;
	uchar components[4];
	uint bits;
} c;

inline void my_memset(void *ptr,int x,int no_of_bytes) {
	unsigned char c;

	while(no_of_bytes--) {
		c = (unsigned char) x;
		*((char *)ptr ) = c ;
		(char *)ptr++;
	}  
}

inline void my_memcpy(void *dest, void *src, size_t n) {
	// Typecast src and dest addresses to (char *)
	char *csrc = (char *)src;
	char *cdest = (char *)dest;

	// Copy contents of src[] to dest[]
	for (int i=0; i<n; i++)
		cdest[i] = csrc[i];
}

inline int clamp_int(int val, int min, int max) {
	return val  max ? max : val);
}

inline uchar clamp_uchar(uchar val, uchar min, uchar max) {
	return val  max ? max : val);
}

inline uchar round_to_5_bits(float val) {
	return clamp_uchar(val * 31.0f / 255.0f + 0.5f, 0, 31);
}

inline uchar round_to_4_bits(float val) {
	return clamp_uchar(val * 15.0f / 255.0f + 0.5f, 0, 15);
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
__constant const uchar g_mod_to_pix[4] = {3, 2, 0, 1};

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
__constant const uchar g_idx_to_num[4][8] = {
	{0, 4, 1, 5, 2, 6, 3, 7},        // Vertical block 0.
	{8, 12, 9, 13, 10, 14, 11, 15},  // Vertical block 1.
	{0, 4, 8, 12, 1, 5, 9, 13},      // Horizontal block 0.
	{2, 6, 10, 14, 3, 7, 11, 15}     // Horizontal block 1.
};

// Constructs a struct Color from a given base struct Color and luminance value.
inline struct Color (__constant struct Color& base, short lum) {
	int b = (int)(base.channels.b) + lum;
	int g = (int)(base.channels.g) + lum;
	int r = (int)(base.channels.r) + lum;
	struct Color struct Color;
	struct Color.channels.b = (uchar)clamp_int(b, 0, 255);
	struct Color.channels.g = (uchar)clamp_int(g, 0, 255);
	struct Color.channels.r = (uchar)clamp_int(r, 0, 255);
	return struct Color;
}

// Calculates the error metric for two struct Colors. A small error signals that the
// struct Colors are similar to each other, a large error the signals the opposite.
inline uint getstruct(struct Color& u, struct Color& v) {
#ifdef USE_PERCEIVED_ERROR_METRIC
	float delta_b = (float)(u.channels.b) - (float)v.channels.b;
	float delta_g = (float)(u.channels.g) - (float)v.channels.g;
	float delta_r = (float)(u.channels.r) - (float)v.channels.r;
	return (uint)(0.299f * delta_b * delta_b +
								 0.587f * delta_g * delta_g +
								 0.114f * delta_r * delta_r);
#else
	int delta_b = (int)(u.channels.b) - (float)v.channels.b;
	int delta_g = (int)(u.channels.g) - (float)v.channels.g;
	int delta_r = (int)(u.channels.r) - (float)v.channels.r;
	return delta_b * delta_b + delta_g * delta_g + delta_r * delta_r;
#endif
}

inline void Writestruct(uchar* block,
						   const struct Color& struct Color0,
						   const struct Color& struct Color1) {
	// Write output struct Color for BGRA textures.
	block[0] = (struct Color0.channels.r & 0xf0) | (struct Color1.channels.r >> 4);
	block[1] = (struct Color0.channels.g & 0xf0) | (struct Color1.channels.g >> 4);
	block[2] = (struct Color0.channels.b & 0xf0) | (struct Color1.channels.b >> 4);
}

inline void Writestruct Colors555(uchar* block,
						   const struct Color& struct Color0,
						   const struct Color& struct Color1) {
	// Table for conversion to 3-bit two complement format.
	static const uchar two_compl_trans_table[8] = {
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
	(short)(struct Color1.channels.r >> 3) - (struct Color0.channels.r >> 3);
	short delta_g =
	(short)(struct Color1.channels.g >> 3) - (struct Color0.channels.g >> 3);
	short delta_b =
	(short)(struct Color1.channels.b >> 3) - (struct Color0.channels.b >> 3);
	
	// Write output struct Color for BGRA textures.
	block[0] = (struct Color0.channels.r & 0xf8) | two_compl_trans_table[delta_r + 4];
	block[1] = (struct Color0.channels.g & 0xf8) | two_compl_trans_table[delta_g + 4];
	block[2] = (struct Color0.channels.b & 0xf8) | two_compl_trans_table[delta_b + 4];
}

inline void WriteCodewordTable(uchar* block,
							   uchar sub_block_id,
							   uchar table) {
	
	uchar shift = (2 + (3 - sub_block_id * 3));
	block[3] &= ~(0x07 << shift);
	block[3] |= table << shift;
}

inline void WritePixelData(uchar* block, uint pixel_data) {
	block[4] |= pixel_data >> 24;
	block[5] |= (pixel_data >> 16) & 0xff;
	block[6] |= (pixel_data >> 8) & 0xff;
	block[7] |= pixel_data & 0xff;
}

inline void WriteFlip(uchar* block, bool flip) {
	block[3] &= ~0x01;
	block[3] |= (uchar)(flip);
}

inline void WriteDiff(uchar* block, bool diff) {
	block[3] &= ~0x02;
	block[3] |= (uchar)(diff) << 1;
}

inline void ExtractBlock(uchar* dst, const uchar* src, int width) {
	for (int j = 0; j < 4; ++j) {
		my_memcpy(&dst[j * 4 * 4], src, 4 * 4);
		src += width * 4;
	}
}

// Compress and rounds BGR888 into BGR444. The resulting BGR444 struct Color is
// expanded to BGR888 as it would be in hardware after decompression. The
// actual 444-bit data is available in the four most significant bits of each
// channel.
inline struct Color makestruct Color444(const float* bgr) {
	uchar b4 = round_to_4_bits(bgr[0]);
	uchar g4 = round_to_4_bits(bgr[1]);
	uchar r4 = round_to_4_bits(bgr[2]);
	struct Color bgr444;
	bgr444.channels.b = (b4 << 4) | b4;
	bgr444.channels.g = (g4 << 4) | g4;
	bgr444.channels.r = (r4 << 4) | r4;
	// Added to distinguish between expanded 555 and 444 struct Colors.
	bgr444.channels.a = 0x44;
	return bgr444;
}

// Compress and rounds BGR888 into BGR555. The resulting BGR555 struct Color is
// expanded to BGR888 as it would be in hardware after decompression. The
// actual 555-bit data is available in the five most significant bits of each
// channel.
inline struct Color makestruct Color555(const float* bgr) {
	uchar b5 = round_to_5_bits(bgr[0]);
	uchar g5 = round_to_5_bits(bgr[1]);
	uchar r5 = round_to_5_bits(bgr[2]);
	struct Color bgr555;
	bgr555.channels.b = (b5 > 2);
	bgr555.channels.g = (g5 > 2);
	bgr555.channels.r = (r5 > 2);
	// Added to distinguish between expanded 555 and 444 struct Colors.
	bgr555.channels.a = 0x55;
	return bgr555;
}
	
void getAveragestruct Color(const struct Color* src, float* avg_struct Color)
{
	uint sum_b = 0, sum_g = 0, sum_r = 0;
	
	for (unsigned int i = 0; i < 8; ++i) {
		sum_b += src[i].channels.b;
		sum_g += src[i].channels.g;
		sum_r += src[i].channels.r;
	}
	
	const float kInv8 = 1.0f / 8.0f;
	avg_struct Color[0] = (float)(sum_b) * kInv8;
	avg_struct Color[1] = (float)(sum_g) * kInv8;
	avg_struct Color[2] = (float)(sum_r) * kInv8;
}
	
unsigned long computeLuminance(uchar* block,
						   const struct Color* src,
						   const struct Color& base,
						   int sub_block_id,
						   const uchar* idx_to_num_tab,
						   unsigned long threshold)
{
	uint best_tbl_err = threshold;
	uchar best_tbl_idx = 0;
	uchar best_mod_idx[8][8];  // [table][texel]

	// Try all codeword tables to find the one giving the best results for this
	// block.
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		// Pre-compute all the candidate struct Colors; combinations of the base struct Color and
		// all available luminance values.
		struct Color candidate_struct Color[4];  // [modifier]
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			candidate_struct Color[mod_idx] = makestruct Color(base, lum);
		}
		
		uint tbl_err = 0;
		
		for (unsigned int i = 0; i < 8; ++i) {
			// Try all modifiers in the current table to find which one gives the
			// smallest error.
			uint best_mod_err = threshold;
			for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
				const struct Color& struct Color = candidate_struct Color[mod_idx];
				
				uint mod_err = getstruct(src[i], struct Color);
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
 * Tries to compress the block under the assumption that it's a single struct Color
 * block. If it's not the function will bail out without writing anything to
 * the destination buffer.
 */
bool tryCompressSolidBlock(uchar* dst,
						   const struct Color* src,
						   unsigned long* error)
{
	for (unsigned int i = 1; i < 16; ++i) {
		if (src[i].bits != src[0].bits)
			return false;
	}
	
	// Clear destination buffer so that we can "or" in the results.
	my_memset(dst, 0, 8);
	
	float src_struct Color_float[3] = {(float)(src->channels.b),
		(float)(src->channels.g),
		(float)(src->channels.r)};
	struct Color base = makestruct Color555(src_struct Color_float);
	
	WriteDiff(dst, true);
	WriteFlip(dst, false);
	Writestruct Colors555(dst, base, base);
	
	uchar best_tbl_idx = 0;
	uchar best_mod_idx = 0;
	uint best_mod_err = UINT32_MAX; 
	
	// Try all codeword tables to find the one giving the best results for this
	// block.
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		// Try all modifiers in the current table to find which one gives the
		// smallest error.
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			const struct Color& struct Color = makestruct Color(base, lum);
			
			uint mod_err = getstruct(*src, struct Color);
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

unsigned long compressBlock(uchar* dst,
												   const struct Color* ver_src,
												   const struct Color* hor_src,
												   unsigned long threshold)
{
	unsigned long solid_error = 0;
	if (tryCompressSolidBlock(dst, ver_src, &solid_error)) {
		return solid_error;
	}
	
	const struct Color* sub_block_src[4] = {ver_src, ver_src + 8, hor_src, hor_src + 8};
	
	struct Color sub_block_avg[4];
	bool use_differential[2] = {true, true};
	
	// Compute the average struct Color for each sub block and determine if differential
	// coding can be used.
	for (unsigned int i = 0, j = 1; i < 4; i += 2, j += 2) {
		float avg_struct Color_0[3];
		getAveragestruct Color(sub_block_src[i], avg_struct Color_0);
		struct Color avg_struct Color_555_0 = makestruct Color555(avg_struct Color_0);
		
		float avg_struct Color_1[3];
		getAveragestruct Color(sub_block_src[j], avg_struct Color_1);
		struct Color avg_struct Color_555_1 = makestruct Color555(avg_struct Color_1);
		
		for (unsigned int light_idx = 0; light_idx < 3; ++light_idx) {
			int u = avg_struct Color_555_0.components[light_idx] >> 3;
			int v = avg_struct Color_555_1.components[light_idx] >> 3;
			
			int component_diff = v - u;
			if (component_diff  3) {
				use_differential[i / 2] = false;
				sub_block_avg[i] = makestruct Color444(avg_struct Color_0);
				sub_block_avg[j] = makestruct Color444(avg_struct Color_1);
			} else {
				sub_block_avg[i] = avg_struct Color_555_0;
				sub_block_avg[j] = avg_struct Color_555_1;
			}
		}
	}
	
	// Compute the error of each sub block before adjusting for luminance. These
	// error values are later used for determining if we should flip the sub
	// block or not.
	uint sub_block_err[4] = {0};
	for (unsigned int i = 0; i < 4; ++i) {
		for (unsigned int j = 0; j < 8; ++j) {
			sub_block_err[i] += getstruct(sub_block_avg[i], sub_block_src[i][j]);
		}
	}
	
	bool flip =
	sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1];
	
	// Clear destination buffer so that we can "or" in the results.
	my_memset(dst, 0, 8);
	
	WriteDiff(dst, use_differential[!!flip]);
	WriteFlip(dst, flip);
	
	uchar sub_block_off_0 = flip ? 2 : 0;
	uchar sub_block_off_1 = sub_block_off_0 + 1;
	
	if (use_differential[!!flip]) {
		Writestruct Colors555(dst, sub_block_avg[sub_block_off_0],
					   sub_block_avg[sub_block_off_1]);
	} else {
		Writestruct (dst, sub_block_avg[sub_block_off_0],
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
	
void compress(const int width, const int height,
				unsigned long compressed_error,
                __global uchar* src,
				__global uchar* dst) {
	
	int x = get_global_id(1);
	struct Color ver_blocks[16];
	struct Color hor_blocks[16];
	//struct Color aux1[8] = { row0, row1, row2, row3,
		row0 + 2, row1 + 2, row2 + 2, row3 + 2};
	
	__constant struct Color* row0 = (__constant struct Color*)(src + x * 4);
	__constant struct Color* row1 = row0 + width;
	__constant struct Color* row2 = row1 + width;
	__constant struct Color* row3 = row2 + width;
			
	my_memcpy(ver_blocks, row0, 8);
	my_memcpy((ver_blocks + 2), row1, 8);
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
			
	compressed_error += compressBlock(dst, ver_blocks, hor_blocks, INT32_MAX);
	
	return compressed_error;
}>>>> file: texture_compress.cpp
#include "compress.hpp"

#include 
#include 
#include 
#include 
#include 

using namespace std;

template 
inline T clamp(T val, T min, T max) {
	return val  max ? max : val);
}

inline uint8_t round_to_5_bits(float val) {
	return clamp(val * 31.0f / 255.0f + 0.5f, 0, 31);
}

inline uint8_t round_to_4_bits(float val) {
	return clamp(val * 15.0f / 255.0f + 0.5f, 0, 15);
}

// Codeword tables.
// See: Table 3.17.2
ALIGNAS(16) static const int16_t g_codeword_tables[8][4] = {
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
static const uint8_t g_mod_to_pix[4] = {3, 2, 0, 1};

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
static const uint8_t g_idx_to_num[4][8] = {
	{0, 4, 1, 5, 2, 6, 3, 7},        // Vertical block 0.
	{8, 12, 9, 13, 10, 14, 11, 15},  // Vertical block 1.
	{0, 4, 8, 12, 1, 5, 9, 13},      // Horizontal block 0.
	{2, 6, 10, 14, 3, 7, 11, 15}     // Horizontal block 1.
};

// Constructs a color from a given base color and luminance value.
inline Color makeColor(const Color& base, int16_t lum) {
	int b = static_cast(base.channels.b) + lum;
	int g = static_cast(base.channels.g) + lum;
	int r = static_cast(base.channels.r) + lum;
	Color color;
	color.channels.b = static_cast(clamp(b, 0, 255));
	color.channels.g = static_cast(clamp(g, 0, 255));
	color.channels.r = static_cast(clamp(r, 0, 255));
	return color;
}

// Calculates the error metric for two colors. A small error signals that the
// colors are similar to each other, a large error the signals the opposite.
inline uint32_t getColorError(const Color& u, const Color& v) {
#ifdef USE_PERCEIVED_ERROR_METRIC
	float delta_b = static_cast(u.channels.b) - v.channels.b;
	float delta_g = static_cast(u.channels.g) - v.channels.g;
	float delta_r = static_cast(u.channels.r) - v.channels.r;
	return static_cast(0.299f * delta_b * delta_b +
								 0.587f * delta_g * delta_g +
								 0.114f * delta_r * delta_r);
#else
	int delta_b = static_cast(u.channels.b) - v.channels.b;
	int delta_g = static_cast(u.channels.g) - v.channels.g;
	int delta_r = static_cast(u.channels.r) - v.channels.r;
	return delta_b * delta_b + delta_g * delta_g + delta_r * delta_r;
#endif
}

inline void WriteColors444(uint8_t* block,
						   const Color& color0,
						   const Color& color1) {
	// Write output color for BGRA textures.
	block[0] = (color0.channels.r & 0xf0) | (color1.channels.r >> 4);
	block[1] = (color0.channels.g & 0xf0) | (color1.channels.g >> 4);
	block[2] = (color0.channels.b & 0xf0) | (color1.channels.b >> 4);
}

inline void WriteColors555(uint8_t* block,
						   const Color& color0,
						   const Color& color1) {
	// Table for conversion to 3-bit two complement format.
	static const uint8_t two_compl_trans_table[8] = {
		4,  // -4 (100b)
		5,  // -3 (101b)
		6,  // -2 (110b)
		7,  // -1 (111b)
		0,  //  0 (000b)
		1,  //  1 (001b)
		2,  //  2 (010b)
		3,  //  3 (011b)
	};
	
	int16_t delta_r =
	static_cast(color1.channels.r >> 3) - (color0.channels.r >> 3);
	int16_t delta_g =
	static_cast(color1.channels.g >> 3) - (color0.channels.g >> 3);
	int16_t delta_b =
	static_cast(color1.channels.b >> 3) - (color0.channels.b >> 3);
	
	// Write output color for BGRA textures.
	block[0] = (color0.channels.r & 0xf8) | two_compl_trans_table[delta_r + 4];
	block[1] = (color0.channels.g & 0xf8) | two_compl_trans_table[delta_g + 4];
	block[2] = (color0.channels.b & 0xf8) | two_compl_trans_table[delta_b + 4];
}

inline void WriteCodewordTable(uint8_t* block,
							   uint8_t sub_block_id,
							   uint8_t table) {
	
	uint8_t shift = (2 + (3 - sub_block_id * 3));
	block[3] &= ~(0x07 << shift);
	block[3] |= table << shift;
}

inline void WritePixelData(uint8_t* block, uint32_t pixel_data) {
	block[4] |= pixel_data >> 24;
	block[5] |= (pixel_data >> 16) & 0xff;
	block[6] |= (pixel_data >> 8) & 0xff;
	block[7] |= pixel_data & 0xff;
}

inline void WriteFlip(uint8_t* block, bool flip) {
	block[3] &= ~0x01;
	block[3] |= static_cast(flip);
}

inline void WriteDiff(uint8_t* block, bool diff) {
	block[3] &= ~0x02;
	block[3] |= static_cast(diff) << 1;
}

inline void ExtractBlock(uint8_t* dst, const uint8_t* src, int width) {
	for (int j = 0; j < 4; ++j) {
		memcpy(&dst[j * 4 * 4], src, 4 * 4);
		src += width * 4;
	}
}

// Compress and rounds BGR888 into BGR444. The resulting BGR444 color is
// expanded to BGR888 as it would be in hardware after decompression. The
// actual 444-bit data is available in the four most significant bits of each
// channel.
inline Color makeColor444(const float* bgr) {
	uint8_t b4 = round_to_4_bits(bgr[0]);
	uint8_t g4 = round_to_4_bits(bgr[1]);
	uint8_t r4 = round_to_4_bits(bgr[2]);
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
	uint8_t b5 = round_to_5_bits(bgr[0]);
	uint8_t g5 = round_to_5_bits(bgr[1]);
	uint8_t r5 = round_to_5_bits(bgr[2]);
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
	uint32_t sum_b = 0, sum_g = 0, sum_r = 0;
	
	for (unsigned int i = 0; i < 8; ++i) {
		sum_b += src[i].channels.b;
		sum_g += src[i].channels.g;
		sum_r += src[i].channels.r;
	}
	
	const float kInv8 = 1.0f / 8.0f;
	avg_color[0] = static_cast(sum_b) * kInv8;
	avg_color[1] = static_cast(sum_g) * kInv8;
	avg_color[2] = static_cast(sum_r) * kInv8;
}
	
unsigned long computeLuminance(uint8_t* block,
						   const Color* src,
						   const Color& base,
						   int sub_block_id,
						   const uint8_t* idx_to_num_tab,
						   unsigned long threshold)
{
	uint32_t best_tbl_err = threshold;
	uint8_t best_tbl_idx = 0;
	uint8_t best_mod_idx[8][8];  // [table][texel]

	// Try all codeword tables to find the one giving the best results for this
	// block.
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		// Pre-compute all the candidate colors; combinations of the base color and
		// all available luminance values.
		Color candidate_color[4];  // [modifier]
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			int16_t lum = g_codeword_tables[tbl_idx][mod_idx];
			candidate_color[mod_idx] = makeColor(base, lum);
		}
		
		uint32_t tbl_err = 0;
		
		for (unsigned int i = 0; i < 8; ++i) {
			// Try all modifiers in the current table to find which one gives the
			// smallest error.
			uint32_t best_mod_err = threshold;
			for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
				const Color& color = candidate_color[mod_idx];
				
				uint32_t mod_err = getColorError(src[i], color);
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

	uint32_t pix_data = 0;

	for (unsigned int i = 0; i < 8; ++i) {
		uint8_t mod_idx = best_mod_idx[best_tbl_idx][i];
		uint8_t pix_idx = g_mod_to_pix[mod_idx];
		
		uint32_t lsb = pix_idx & 0x1;
		uint32_t msb = pix_idx >> 1;
		
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
bool tryCompressSolidBlock(uint8_t* dst,
						   const Color* src,
						   unsigned long* error)
{
	for (unsigned int i = 1; i < 16; ++i) {
		if (src[i].bits != src[0].bits)
			return false;
	}
	
	// Clear destination buffer so that we can "or" in the results.
	memset(dst, 0, 8);
	
	float src_color_float[3] = {static_cast(src->channels.b),
		static_cast(src->channels.g),
		static_cast(src->channels.r)};
	Color base = makeColor555(src_color_float);
	
	WriteDiff(dst, true);
	WriteFlip(dst, false);
	WriteColors555(dst, base, base);
	
	uint8_t best_tbl_idx = 0;
	uint8_t best_mod_idx = 0;
	uint32_t best_mod_err = UINT32_MAX; 
	
	// Try all codeword tables to find the one giving the best results for this
	// block.
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		// Try all modifiers in the current table to find which one gives the
		// smallest error.
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			int16_t lum = g_codeword_tables[tbl_idx][mod_idx];
			const Color& color = makeColor(base, lum);
			
			uint32_t mod_err = getColorError(*src, color);
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
	
	uint8_t pix_idx = g_mod_to_pix[best_mod_idx];
	uint32_t lsb = pix_idx & 0x1;
	uint32_t msb = pix_idx >> 1;
	
	uint32_t pix_data = 0;
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

TextureCompressor::TextureCompressor() { } 	// constructor, CPU, nothing to do
TextureCompressor::~TextureCompressor() { }	// destructor, CPU, nothing to do

unsigned long TextureCompressor::compressBlock(uint8_t* dst,
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
	uint32_t sub_block_err[4] = {0};
	for (unsigned int i = 0; i < 4; ++i) {
		for (unsigned int j = 0; j < 8; ++j) {
			sub_block_err[i] += getColorError(sub_block_avg[i], sub_block_src[i][j]);
		}
	}
	
	bool flip =
	sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1];
	
	// Clear destination buffer so that we can "or" in the results.
	memset(dst, 0, 8);
	
	WriteDiff(dst, use_differential[!!flip]);
	WriteFlip(dst, flip);
	
	uint8_t sub_block_off_0 = flip ? 2 : 0;
	uint8_t sub_block_off_1 = sub_block_off_0 + 1;
	
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
	
unsigned long TextureCompressor::compress(const uint8_t* src,
											  uint8_t* dst,
											  int width,
											  int height) {
	Color ver_blocks[16];
	Color hor_blocks[16];
	
	unsigned long compressed_error = 0;
	
	for (int y = 0; y < height; y += 4, src += width * 4 * 4) {
		for (int x = 0; x < width; x += 4, dst += 8) {
			const Color* row0 = reinterpret_cast(src + x * 4);
			const Color* row1 = row0 + width;
			const Color* row2 = row1 + width;
			const Color* row3 = row2 + width;
			
			memcpy(ver_blocks, row0, 8);
			memcpy(ver_blocks + 2, row1, 8);
			memcpy(ver_blocks + 4, row2, 8);
			memcpy(ver_blocks + 6, row3, 8);
			memcpy(ver_blocks + 8, row0 + 2, 8);
			memcpy(ver_blocks + 10, row1 + 2, 8);
			memcpy(ver_blocks + 12, row2 + 2, 8);
			memcpy(ver_blocks + 14, row3 + 2, 8);
			
			memcpy(hor_blocks, row0, 16);
			memcpy(hor_blocks + 4, row1, 16);
			memcpy(hor_blocks + 8, row2, 16);
			memcpy(hor_blocks + 12, row3, 16);
			
			compressed_error += compressBlock(dst, ver_blocks, hor_blocks, INT32_MAX);
		}
	}
	
	return compressed_error;
}
>>>> file: texture_compress_skl.cpp
#include "compress.hpp"

#include 
#include 
#include 
#include 
#include 

using namespace std;

TextureCompressor::TextureCompressor() { 
	gpu_find(this->device);
} 	// constructor/Users/grigore.lupescu/Desktop/RESEARCH/asc/teme/tema3/2018/Tema3-schelet/src/compress.cpp
TextureCompressor::~TextureCompressor() { }	// destructor

void TextureCompressor::gpu_find(cl_device_id &device)
{


	cl_platform_id platform;
	cl_uint platform_num = 0;
	
	cl_uint device_num = 0;
	
	size_t attr_size = 0;
	cl_char* attr_data = NULL;
	
	/* get num of available OpenCL platforms */
	CL_ERR( clGetPlatformIDs(0, NULL, &platform_num));
	this->platform_ids = new cl_platform_id[platform_num];
	DIE(this->platform_ids == NULL, "alloc this->platform_ids");
	
	/* get all available OpenCL platforms */
	CL_ERR( clGetPlatformIDs(platform_num, this->platform_ids, NULL));
	cout << "Platforms found: " << platform_num << endl;
	
	/* list all platforms and VENDOR/VERSION properties */
	for(uint platf=0; platf<platform_num; platf++)
	{
		/* get attribute CL_PLATFORM_VENDOR */
		CL_ERR( clGetPlatformInfo(this->platform_ids[platf],
								  CL_PLATFORM_VENDOR, 0, NULL, &attr_size));
		attr_data = new cl_char[attr_size];
		DIE(attr_data == NULL, "alloc attr_data");
		
		/* get data CL_PLATFORM_VENDOR */
		CL_ERR( clGetPlatformInfo(this->platform_ids[platf],
								  CL_PLATFORM_VENDOR, attr_size, attr_data, NULL));
		cout << "Platform " << platf << " " << attr_data << " ";
		delete[] attr_data;
		
		/* get attribute size CL_PLATFORM_VERSION */
		CL_ERR( clGetPlatformInfo(this->platform_ids[platf],
								  CL_PLATFORM_VERSION, 0, NULL, &attr_size));
		attr_data = new cl_char[attr_size];
		DIE(attr_data == NULL, "alloc attr_data");
		
		/* get data size CL_PLATFORM_VERSION */
		CL_ERR( clGetPlatformInfo(this->platform_ids[platf],
								  CL_PLATFORM_VERSION, attr_size, attr_data, NULL));
		cout << attr_data << endl;
		delete[] attr_data;
		
		/* no valid platform found */
		platform = this->platform_ids[platf];
		DIE(platform == 0, "platform selection");
		
		/* get num of available OpenCL devices type GPU on the selected platform */
		CL_ERR( clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &device_num));
		this->device_ids = new cl_device_id[device_num];
		DIE(this->device_ids == NULL, "alloc devices");
		
		/* get all available OpenCL devices type GPU on the selected platform */
		CL_ERR( clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL,
							   device_num, this->device_ids, NULL));
		cout << "\tDevices found " << device_num  << endl;
		
		/* list all devices and TYPE/VERSION properties */
		for(uint dev=0; dev<device_num; dev++)
		{
			/* get attribute size */
			CL_ERR( clGetDeviceInfo(this->device_ids[dev], CL_DEVICE_NAME,
									0, NULL, &attr_size));
			attr_data = new cl_char[attr_size];
			DIE(attr_data == NULL, "alloc attr_data");
			
			/* get attribute CL_DEVICE_NAME */
			CL_ERR( clGetDeviceInfo(this->device_ids[dev], CL_DEVICE_NAME,
									attr_size, attr_data, NULL));
			cout << "\tDevice " << dev << " " << attr_data << " ";
			delete[] attr_data;
			
			/* get attribute size */
			CL_ERR( clGetDeviceInfo(this->device_ids[dev], CL_DEVICE_VERSION,
									0, NULL, &attr_size));
			attr_data = new cl_char[attr_size];
			DIE(attr_data == NULL, "alloc attr_data");
			
			/* get attribute CL_DEVICE_VERSION */
			CL_ERR( clGetDeviceInfo(this->device_ids[dev], CL_DEVICE_VERSION,
									attr_size, attr_data, NULL));
			cout << attr_data;
			delete[] attr_data;

			cl_device_type devType;
			clGetDeviceInfo(this->device_ids[dev], CL_DEVICE_TYPE,
            	sizeof(cl_device_type), &devType, NULL);
			
			/* select device based on cli arguments */
			if(devType == CL_DEVICE_TYPE_GPU){
				device = this->device_ids[dev];
				cout << " <--- SELECTED ";
			}
			
			cout << endl;
		}
	}
	
	delete[] this->platform_ids;
	delete[] this->device_ids;
}


unsigned long TextureCompressor::compress(const uint8_t* src,
									  uint8_t* dst,
									  int width,
									  int height)
{	
	int				szA, szB;
	string			kernel_src;
	cl_uint			nd;
	cl_mem			a_in;
	cl_mem			b_out;
	int				i, ret;


	unsigned int    compressed_error;
	/* create a context for the device */
	this->context = clCreateContext(0, 1, &this->device, NULL, NULL, &ret);
	CL_ERR( ret );

	this->command_queue = clCreateCommandQueue(this->context , this->device,
									CL_QUEUE_PROFILING_ENABLE, &ret);
	CL_ERR( ret );
	


	szA = szB = (width * height);

	a_in   = clCreateBuffer(context,  CL_MEM_READ_ONLY,
							sizeof(uint8_t) * szA, NULL, NULL);
	b_out   = clCreateBuffer(context,  CL_MEM_WRITE_ONLY,
							sizeof(uint8_t) * szB, NULL, NULL);

	read_kernel("sol_device.cl", kernel_src);
	const char* kernel_c_str = kernel_src.c_str();



	this->program = clCreateProgramWithSource(this->context, 1,
				(const char **) &kernel_c_str, NULL, &ret);
	CL_ERR( ret );

	ret = clBuildProgram(this->program, 0, NULL, NULL, NULL, NULL);
	CL_COMPILE_ERR( ret, this->program, this->device);
	
	// Create the compute kernel from the program
	this->kernel = clCreateKernel(this->program, "mmul", &ret);
	CL_ERR( ret );

	ret  = 0;
	ret  = clSetKernelArg(kernel, 0, sizeof(int), &width);
	ret  = clSetKernelArg(kernel, 1, sizeof(int), &height);


	ret  = clSetKernelArg(kernel, 2, sizeof(unsigned long), &compressed_error);
	ret |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &a_in);
	ret |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &b_out);

	ret = clEnqueueWriteBuffer(this->command_queue, a_in, CL_TRUE, 0,
							   sizeof(uint8_t) * szA, src, 0, NULL, NULL);
	
	cl_event prof_event;


	ret = clEnqueueNDRangeKernel(this->command_queue, this->kernel, 2, NULL,
								 NULL, NULL, 0, NULL, &prof_event);

	clFinish(this->command_queue);

	ret = clEnqueueReadBuffer( this->command_queue, b_out, CL_TRUE, 0,
							  sizeof(uint8_t) * szB, dst, 0, NULL, NULL );

	clReleaseProgram(this->program);
	clReleaseKernel(this->kernel);
	clReleaseMemObject(a_in);
	clReleaseMemObject(b_out);
	clReleaseCommandQueue(this->command_queue);
	clReleaseContext(this->context);
}
