
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

uchar wrapper_clamp(int val, uchar min, uchar max){
	return val  max ? max : val);
}



uchar wrapper_clamp2(uchar val, uchar min, uchar max){
	return val  max ? max : val);
}


inline uchar round_to_5_bits(float val) {
	return wrapper_clamp2((uchar)(val * 31.0f / 255.0f + 0.5f), (uchar)0, (uchar)31);
}

inline uchar round_to_4_bits(float val) {
	return wrapper_clamp2((uchar)(val * 15.0f / 255.0f + 0.5f),(uchar) 0,(uchar) 15);
}

inline union Color makeColor(union Color base, short lum) {
	int b = (int)((int)(base.channels.b) + lum);
	int g = (int)((int)(base.channels.g) + lum);
	int r = (int)((int)(base.channels.r) + lum);
	union Color color;
	color.channels.b = (uchar)(wrapper_clamp(b, 0, 255));
	color.channels.g = (uchar)(wrapper_clamp(g, 0, 255));
	color.channels.r = (uchar)(wrapper_clamp(r, 0, 255));
	return color;
}

inline void WriteColors444(__global uchar* block,
						   union Color color0,
						   union Color color1) {
	
	block[0] = (color0.channels.r & 0xf0) | (color1.channels.r >> 4);
	block[1] = (color0.channels.g & 0xf0) | (color1.channels.g >> 4);
	block[2] = (color0.channels.b & 0xf0) | (color1.channels.b >> 4);
}


inline void WriteColors555(__global uchar* block,
						   union Color color0,
						   union Color color1) {
	
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

	short delta_r =	(short)((color1.channels.r >> 3) - (color0.channels.r >> 3));
	short delta_g = (short) ((color1.channels.g >> 3) - (color0.channels.g >> 3));
	short delta_b = (short) ((color1.channels.b >> 3) - (color0.channels.b >> 3));

	
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

inline void ExtractBlock(uchar* dst, const uchar* src, int width) {
	for (int j = 0; j < 4; ++j) {
		
		for (int k = 0; k < 16; k++) {
			dst[j * 16 + k ] = *(src + k);
		}


		src += width * 4;
	}
}


inline union Color makeColor444(const float* bgr) {
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


inline union Color makeColor555(const float* bgr) {
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

inline uint getColorError(union Color u, union Color v) {
#ifdef USE_PERCEIVED_ERROR_METRIC
	float delta_b = (float)((u.channels.b) - v.channels.b);
	float delta_g = (float)((u.channels.g) - v.channels.g);
	float delta_r = (float)((u.channels.r) - v.channels.r);
	return (uint)(0.299f * delta_b * delta_b +
								 0.587f * delta_g * delta_g +
								 0.114f * delta_r * delta_r);
#else
	int delta_b = (int)((u.channels.b) - v.channels.b);
	int delta_g = (int)((u.channels.g) - v.channels.g);
	int delta_r = (int)((u.channels.r) - v.channels.r);
	return delta_b * delta_b + delta_g * delta_g + delta_r * delta_r;
#endif
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
	avg_color[0] = (float)((sum_b) * kInv8);


	avg_color[1] = (float)((sum_g) * kInv8);
	avg_color[2] = (float)((sum_r) * kInv8);
}

unsigned long computeLuminance(__global uchar* block,
						   const union Color* src,


						   union Color base,
						   int sub_block_id,
						   const uchar* idx_to_num_tab,
						   unsigned long threshold)
{
	uint best_tbl_err = threshold;
	uchar best_tbl_idx = 0;
	uchar best_mod_idx[8][8];  

	const short g_codeword_tables[8][4] __attribute__((aligned(16))) = {
		{-8, -2, 2, 8},
		{-17, -5, 5, 17},
		{-29, -9, 9, 29},
		{-42, -13, 13, 42},
		{-60, -18, 18, 60},
		{-80, -24, 24, 80},
		{-106, -33, 33, 106},
		{-183, -47, 47, 183}};



	const uchar g_mod_to_pix[4] = {3, 2, 0, 1};



	
	
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
				union Color color = candidate_color[mod_idx];

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

bool tryCompressSolidBlock(__global uchar* dst,
						   const union Color* src,
						   unsigned long* error)
{
	const uchar g_mod_to_pix[4] = {3, 2, 0, 1};

	const short g_codeword_tables[8][4] __attribute__((aligned(16))) = {
	{-8, -2, 2, 8},
	{-17, -5, 5, 17},
	{-29, -9, 9, 29},
	{-42, -13, 13, 42},
	{-60, -18, 18, 60},
	{-80, -24, 24, 80},
	{-106, -33, 33, 106},
	{-183, -47, 47, 183}};

	const uchar g_idx_to_num[4][8] = {
	{0, 4, 1, 5, 2, 6, 3, 7},        
	{8, 12, 9, 13, 10, 14, 11, 15},  
	{0, 4, 8, 12, 1, 5, 9, 13},      
	{2, 6, 10, 14, 3, 7, 11, 15}     
};

	for (unsigned int i = 1; i < 16; ++i) {
		if (src[i].bits != src[0].bits)
			return false;
	}

	
	
	for (int i = 0; i < 8; i++) {
		dst[i] = 0;
	}

	float src_color_float[3] = {(float)(src->channels.b), (float)(src->channels.g),
				(float)(src->channels.r)};
	union Color base = makeColor555(src_color_float);

	WriteDiff(dst, true);
	WriteFlip(dst, false);
	WriteColors555(dst, base, base);

	uchar best_tbl_idx = 0;
	uchar best_mod_idx = 0;
	uint best_mod_err = 0x7fffffff;

	
	
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		
		


		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			const union Color color = makeColor(base, lum);

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
unsigned long compressBlock(__global uchar* dst, const union Color* ver_src, const union Color* hor_src,
												   unsigned long threshold)
{
		uchar g_idx_to_num[4][8] = {
			{0, 4, 1, 5, 2, 6, 3, 7},        
			{8, 12, 9, 13, 10, 14, 11, 15},  
			{0, 4, 8, 12, 1, 5, 9, 13},      
			{2, 6, 10, 14, 3, 7, 11, 15}     
		};

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
			sub_block_err[i] += getColorError(sub_block_avg[i], sub_block_src[i][j]);
		}
	}

	bool flip =
	sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1];

	
	
	for (int i  = 0; i < 8; i++)
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

	unsigned long lumi_error1 = 0, lumi_error2 = 0;

	
	lumi_error1 = computeLuminance(dst, sub_block_src[sub_block_off_0],
								   sub_block_avg[sub_block_off_0], 0,
								   g_idx_to_num[sub_block_off_0],
								   threshold);
	
	lumi_error2 = computeLuminance(dst, sub_block_src[sub_block_off_1],
								   sub_block_avg[sub_block_off_1], 1,
								   g_idx_to_num[sub_block_off_1],
								   threshold);

		return lumi_error1 + lumi_error2;
}

__kernel void compress_k(const int width, const int height, __global uchar *src,
				__global uchar *dst)
{
	int y = get_global_id(0);
	int x = get_global_id(1);


	union Color ver_blocks[16]; 
	union Color hor_blocks[16];

	__local union Color row0[4]; 
	__local union Color row1[4];
	__local union Color row2[4];
	__local union Color row3[4];

	int depl_src = y * 16 * width + (x * 16);

	for (int i  = 0; i < 4; i++) {
		int depl = depl_src; 
		row0[i].channels.b = src[depl + (i * 4)];
		row0[i].channels.g = src[depl + (i * 4) + 1];
		row0[i].channels.r = src[depl + (i * 4) + 2];
		row0[i].channels.a = src[depl + (i * 4) + 3];
	}

	for (int i  = 0; i < 4; i++) {
		int depl = depl_src + (width * 4) + (i * 4);
		row1[i].channels.b = src[depl];
		row1[i].channels.g = src[depl + 1];
		row1[i].channels.r = src[depl + 2];
		row1[i].channels.a = src[depl + 3];
	}

	for (int i  = 0; i < 4; i++) {
		int depl = depl_src + 2 * (width * 4) + (i * 4);
		row2[i].channels.b = src[depl];
		row2[i].channels.g = src[depl + 1];
		row2[i].channels.r = src[depl + 2];
		row2[i].channels.a = src[depl + 3];
	}

	for (int i  = 0; i < 4; i++) {
		int depl = depl_src + 3 * (width * 4) + (i * 4);
		row3[i].channels.b = src[depl];
		row3[i].channels.g = src[depl + 1];
		row3[i].channels.r = src[depl + 2];
		row3[i].channels.a = src[depl + 3];
	}

	
	for (int i = 0; i < 2; i++) {
		ver_blocks[i].channels.b = row0[i].channels.b;
		ver_blocks[i].channels.g = row0[i].channels.g;
		ver_blocks[i].channels.r = row0[i].channels.r;
		ver_blocks[i].channels.a = row0[i].channels.a;
		ver_blocks[2 + i].channels.b = row1[i].channels.b;
		ver_blocks[2 + i].channels.g = row1[i].channels.g;
		ver_blocks[2 + i].channels.r = row1[i].channels.r;
		ver_blocks[2 + i].channels.a = row1[i].channels.a;
		ver_blocks[4 + i].channels.b = row2[i].channels.b;
		ver_blocks[4 + i].channels.g = row2[i].channels.g;
		ver_blocks[4 + i].channels.r = row2[i].channels.r;
		ver_blocks[4 + i].channels.a = row2[i].channels.a;
		ver_blocks[6 + i].channels.b = row3[i].channels.b;
		ver_blocks[6 + i].channels.g = row3[i].channels.g;
		ver_blocks[6 + i].channels.r = row3[i].channels.r;
		ver_blocks[6 + i].channels.a = row3[i].channels.a;

		ver_blocks[8 + i].channels.b = row0[2 + i].channels.b;
		ver_blocks[8 + i].channels.g = row0[2 + i].channels.g;
		ver_blocks[8 + i].channels.r = row0[2 + i].channels.r;
		ver_blocks[8 + i].channels.a = row0[2 + i].channels.a;
		ver_blocks[10 + i].channels.b = row1[2 + i].channels.b;
		ver_blocks[10 + i].channels.g = row1[2 + i].channels.g;
		ver_blocks[10 + i].channels.r = row1[2 + i].channels.r;
		ver_blocks[10 + i].channels.a = row1[2 + i].channels.a;
		ver_blocks[12 + i].channels.b = row2[2 + i].channels.b;
		ver_blocks[12 + i].channels.g = row2[2 + i].channels.g;
		ver_blocks[12 + i].channels.r = row2[2 + i].channels.r;
		ver_blocks[12 + i].channels.a = row2[2 + i].channels.a;
		ver_blocks[14 + i].channels.b = row3[2 + i].channels.b;
		ver_blocks[14 + i].channels.g = row3[2 + i].channels.g;
		ver_blocks[14 + i].channels.r = row3[2 + i].channels.r;
		ver_blocks[14 + i].channels.a = row3[2 + i].channels.a;
	}

	
	for (int i = 0; i < 4; i++) {
		ver_blocks[i].channels.b = row0[i].channels.b;
		ver_blocks[i].channels.g = row0[i].channels.g;
		ver_blocks[i].channels.r = row0[i].channels.r;
		ver_blocks[i].channels.a = row0[i].channels.a;
		ver_blocks[4 + i].channels.b = row1[i].channels.b;
		ver_blocks[4 + i].channels.g = row1[i].channels.g;
		ver_blocks[4 + i].channels.r = row1[i].channels.r;
		ver_blocks[4 + i].channels.a = row1[i].channels.a;
		ver_blocks[8 + i].channels.b = row2[i].channels.b;
		ver_blocks[8 + i].channels.g = row2[i].channels.g;
		ver_blocks[8 + i].channels.r = row2[i].channels.r;
		ver_blocks[8 + i].channels.a = row2[i].channels.a;
		ver_blocks[12 + i].channels.b = row3[i].channels.b;
		ver_blocks[12 + i].channels.g = row3[i].channels.g;
		ver_blocks[12 + i].channels.r = row3[i].channels.r;
		ver_blocks[12 + i].channels.a = row3[i].channels.a;

	}
	int dst_depl = (y * (width / 4) + x) * 8;
	compressBlock(dst + dst_depl, ver_blocks, hor_blocks, 0xffffffff);


}
#include "compress.hpp"

#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include 


static int number_of_devices;

#define DIE(assertion, call_description)                    \
do {                                                        \
    if (assertion) {                                        \
            fprintf(stderr, "(%d): ",                       \
                            __LINE__);                      \
            perror(call_description);                       \
            exit(EXIT_FAILURE);                             \
    }                                                       \
} while(0);

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


void cl_get_compiler_err_log(cl_program program,
                             cl_device_id device)
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


int CL_COMPILE_ERR(int cl_ret,
                  cl_program program,
                  cl_device_id device, int line)
{
	if(cl_ret != CL_SUCCESS){
		cout << endl << cl_get_string_err(cl_ret) << " (" << line << ")"<< endl;
		cl_get_compiler_err_log(program, device);
		return 1;
	}
	return 0;
}

int CL_ERR(int cl_ret, int line)
{
	if(cl_ret != CL_SUCCESS){
		cout << endl << cl_get_string_err(cl_ret) << " (" << line << ")" << endl;
		return 1;
	}
	return 0;
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

static pair, vector> gpu_find()
{

	vector device_idsv;
	vector platform_idsv;

	cl_platform_id platform;
	cl_uint platform_num = 0;
	cl_platform_id* platform_list = NULL;

	cl_uint device_num = 0;
	cl_device_id* device_list = NULL;

	size_t attr_size = 0;
	cl_char* attr_data = NULL;

	
	CL_ERR( clGetPlatformIDs(0, NULL, &platform_num), __LINE__);
	platform_list = new cl_platform_id[platform_num];
	DIE(platform_list == NULL, "alloc platform_list");

	
	CL_ERR( clGetPlatformIDs(platform_num, platform_list, NULL), __LINE__);
	cout << "Platforms found: " << platform_num << endl;

	
	for(uint platf=0; platf<platform_num; platf++)
	{
		platform_idsv.push_back(platform_list[platf]);
		
		CL_ERR( clGetPlatformInfo(platform_list[platf],
								  CL_PLATFORM_VENDOR, 0, NULL, &attr_size), __LINE__);
		attr_data = new cl_char[attr_size];
		DIE(attr_data == NULL, "alloc attr_data");

		
		CL_ERR( clGetPlatformInfo(platform_list[platf],
								  CL_PLATFORM_VENDOR, attr_size, attr_data, NULL), __LINE__);
		cout << "Platform " << platf << " " << attr_data << " ";
		delete[] attr_data;

		
		CL_ERR( clGetPlatformInfo(platform_list[platf],
								  CL_PLATFORM_VERSION, 0, NULL, &attr_size), __LINE__);
		attr_data = new cl_char[attr_size];
		DIE(attr_data == NULL, "alloc attr_data");

		
		CL_ERR( clGetPlatformInfo(platform_list[platf],
								  CL_PLATFORM_VERSION, attr_size, attr_data, NULL), __LINE__);
		cout << attr_data << endl;
		delete[] attr_data;

		
		platform = platform_list[platf];
		DIE(platform == 0, "platform selection");

        cl_int ret;
		
		ret =  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &device_num);
        if (ret != CL_SUCCESS) {
            cout << "\tNo GPU devices found on this platform" << endl;
            continue;
        }
		device_list = new cl_device_id[device_num];
		DIE(device_list == NULL, "alloc devices");

		
		ret = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU,
							   device_num, device_list, NULL);
        if (ret != CL_SUCCESS) {
            cout << "No GPU devices found on this platform" << endl;
            continue;
        }
		cout << "\tDevices found " << device_num  << endl;

		
		for(uint dev=0; dev<device_num; dev++)
		{
			device_idsv.push_back(device_list[dev]);
			
			CL_ERR( clGetDeviceInfo(device_list[dev], CL_DEVICE_NAME,
									0, NULL, &attr_size), __LINE__);
			attr_data = new cl_char[attr_size];
			DIE(attr_data == NULL, "alloc attr_data");

			
			CL_ERR( clGetDeviceInfo(device_list[dev], CL_DEVICE_NAME,
									attr_size, attr_data, NULL), __LINE__);
			cout << "\tDevice " << dev << " " << attr_data << " ";
			delete[] attr_data;

			
			CL_ERR( clGetDeviceInfo(device_list[dev], CL_DEVICE_VERSION,
									0, NULL, &attr_size), __LINE__);
			attr_data = new cl_char[attr_size];
			DIE(attr_data == NULL, "alloc attr_data");

			
			CL_ERR( clGetDeviceInfo(device_list[dev], CL_DEVICE_VERSION,
									attr_size, attr_data, NULL), __LINE__);
			cout << attr_data;
			delete[] attr_data;

			cout << endl;
		}
	}

	delete[] platform_list;
	delete[] device_list;

	return make_pair(platform_idsv, device_idsv);
}

TextureCompressor::TextureCompressor() {
	pair, vector> res = gpu_find();

	platform_ids =  new cl_platform_id[res.first.size()];
	DIE(platform_ids == NULL, "platform_ids");
	device_ids =  new cl_device_id[res.second.size()];
	DIE(device_ids == NULL, "device_ids");

	for (int i  = 0; i < res.second.size(); i++) {
		device_ids[i] = res.second[i];
	}

	for (int i  = 0; i < res.first.size(); i++) {
		platform_ids[i] = res.first[i];
	}

	device = device_ids[0];
    number_of_devices = res.second.size();
    cout << number_of_devices << " devices in list" <<endl;
}
TextureCompressor::~TextureCompressor() {

	delete[] device_ids;
	delete[] platform_ids;
}	

unsigned long TextureCompressor::compress(const uint8_t* src,
									  uint8_t* dst,
									  int width,
									  int height)
{
	cout << "Width: " << width << endl;
	cout << "Height: " << height << endl;
	cl_mem src_dev;
	cl_mem dst_dev;

	size_t global[2];

	int ret;
	string kernel_src;
	
	context = clCreateContext(0, 1, &device, NULL, NULL, &ret);
	CL_ERR( ret, __LINE__);

	command_queue = clCreateCommandQueue(context, device, 0, &ret);
	CL_ERR( ret ,__LINE__ );

	int sz_src = width * height * 4;
	int sz_dst = sz_src / 8;

	
	
	
	
	src_dev = clCreateBuffer(context,  CL_MEM_READ_ONLY,
							sizeof(uint8_t) * sz_src, NULL, &ret);
    CL_ERR( ret, __LINE__ );
    dst_dev  = clCreateBuffer(context,  CL_MEM_READ_WRITE,
							sizeof(uint8_t) * sz_dst, NULL, &ret);
    CL_ERR( ret, __LINE__ );

	
	read_kernel("compress_device.cl", kernel_src);
	const char* kernel_c_str = kernel_src.c_str();

	
	program = clCreateProgramWithSource(context, 1,(const char **) &kernel_c_str, NULL, &ret);
	CL_ERR( ret, __LINE__ );

	
	ret = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	CL_COMPILE_ERR( ret, program, device, __LINE__);

	
	kernel = clCreateKernel(program, "compress_k", &ret);
	CL_ERR( ret,__LINE__ );

	
	ret  = 0;
	ret  = clSetKernelArg(kernel, 0, sizeof(int), &width);
	ret |= clSetKernelArg(kernel, 1, sizeof(int), &height);
	ret |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &src_dev);
	ret |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &dst_dev);

	
	ret = clEnqueueWriteBuffer(command_queue, src_dev, CL_TRUE, 0,
							   sizeof(uint8_t) * sz_src, src, 0, NULL, NULL);
	cl_event prof_event;

	global[0] =(size_t) (height / 4);
    global[1] =(size_t) (width / 4);

	ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL,
								 global, NULL, 0, NULL, &prof_event);
    CL_COMPILE_ERR( ret, program, device, __LINE__);
	
	clFinish(command_queue);

	
	ret = clEnqueueReadBuffer(command_queue, dst_dev, CL_TRUE, 0,
							  sizeof(uint8_t) * sz_dst, dst, 0, NULL, NULL );

    clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseMemObject(src_dev);
	clReleaseMemObject(dst_dev);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
	return 0;
}
