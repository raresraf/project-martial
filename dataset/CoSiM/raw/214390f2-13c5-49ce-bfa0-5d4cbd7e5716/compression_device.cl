
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


inline uint my_clamp(int val, int min, int max) {
	if (val < min)
		return min;
	else if (val > max)
		return max;
	return val;
}

inline uchar round_to_5_bits(int val) {
	return (uchar) my_clamp(val * 31.0f / 255.0f + 0.5f, 0, 31);
}

inline uchar round_to_4_bits(int val) {
	return (uchar) my_clamp(val * 15.0f / 255.0f + 0.5f, 0, 15);
}

__constant short g_mod_to_pix[4] = {3, 2, 0, 1};


inline union Color* makeColor(union Color base, short lum) {
	int b = (int)base.channels.b + (int)lum;
	int g = (int)base.channels.g + (int)lum;
	int r = (int)base.channels.r + (int)lum;
	union Color* color;
	color->channels.b = (uchar)(clamp(b, 0, 255));
	color->channels.g = (uchar)(clamp(g, 0, 255));
	color->channels.r = (uchar)(clamp(r, 0, 255));
	return (union Color*) color;
}

inline uint getColorError(union Color u, union Color v) {
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
						    union Color color0,
						    union Color color1
								) {
	
	block[0] = (color0.channels.r & 0xf0) | (color1.channels.r >> 4);
	block[1] = (color0.channels.g & 0xf0) | (color1.channels.g >> 4);
	block[2] = (color0.channels.b & 0xf0) | (color1.channels.b >> 4);
}



inline void WriteColors555(__global uchar* block,
						   union Color color0,
						   union Color color1) {
	
	uchar two_compl_trans_table[8] = {
		4,  
		5,  
		6,  
		7,  
		0,  
		1,  
		2,  
		3,  
	};

	short delta_r =
	(short)(color1.channels.r >> 3) - (color0.channels.r >> 3);
	short delta_g =
	(short)(color1.channels.g >> 3) - (color0.channels.g >> 3);
	short delta_b =
	(short)(color1.channels.b >> 3) - (color0.channels.b >> 3);

	
	block[0] = (color0.channels.r & 0xf8) | two_compl_trans_table[delta_r + 4];
	block[1] = (color0.channels.g & 0xf8) | two_compl_trans_table[delta_g + 4];
	block[2] = (color0.

	channels.b & 0xf8) | two_compl_trans_table[delta_b + 4];
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

inline void memcpy(uchar *dst, uchar *src, int width) {
	for (int i = 0; i < width; i++) {
		dst[i] = src[i];
	}
}

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

void getAverageColor(union Color* src, float* avg_color)
{
	uint sum_b = 0, sum_g = 0, sum_r = 0;

	for (uint i = 0; i < 8; ++i) {
		sum_b += src[i].channels.b;


		sum_g += src[i].channels.g;
		sum_r += src[i].channels.r;
	}

	float kInv8 = 1.0f / 8.0f;
	avg_color[0] = (float)(sum_b) * kInv8;
	avg_color[1] = (float)(sum_g) * kInv8;
	avg_color[2] = (float)(sum_r) * kInv8;
}

void memset(__global uchar* dst, int value, int size) {
	for (int i = 0; i < size; i++) {
		dst[i] = value;
	}
}

unsigned long computeLuminance(__global uchar* block,
						   union Color* src,
						   union Color base,
						   int sub_block_id,
						   uchar* idx_to_num_tab,
						   unsigned long threshold)
{
	short g_codeword_tables[8][4] = {
	{-8, -2, 2, 8},
	{-17, -5, 5, 17},
	{-29, -9, 9, 29},
	{-42, -13, 13, 42},
	{-60, -18, 18, 60},
	{-80, -24, 24, 80},
	{-106, -33, 33, 106},
	{-183, -47, 47, 183}};

	uint best_tbl_err = threshold;
	uchar best_tbl_idx = 0;
	uchar best_mod_idx[8][8];  

	
	
	for (uint tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		
		
		union Color candidate_color[4];  
		for (uint mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			candidate_color[mod_idx] = *makeColor(base, lum);
		}

		uint tbl_err = 0;

		for (uint i = 0; i < 8; ++i) {
			
			
			uint best_mod_err = threshold;


			for (uint mod_idx = 0; mod_idx < 4; ++mod_idx) {
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
						   union Color* src,
						   unsigned long* error)
{

	short g_codeword_tables[8][4] = {
	{-8, -2, 2, 8},
	{-17, -5, 5, 17},
	{-29, -9, 9, 29},
	{-42, -13, 13, 42},
	{-60, -18, 18, 60},
	{-80, -24, 24, 80},
	{-106, -33, 33, 106},
	{-183, -47, 47, 183}};

	uchar g_idx_to_num[4][8] = {
	{0, 4, 1, 5, 2, 6, 3, 7},        
	{8, 12, 9, 13, 10, 14, 11, 15},  
	{0, 4, 8, 12, 1, 5, 9, 13},      
	{2, 6, 10, 14, 3, 7, 11, 15}     
	};

	for (unsigned int i = 1; i < 16; ++i) {
		if (src[i].bits != src[0].bits)
			return false;
	}

	
	memset(dst, 0, 8);

	float src_color_float[3] = {(float)(src->channels.b),
		(float)(src->channels.g),
		(float)(src->channels.r)};
	union Color base = makeColor555(src_color_float);

	WriteDiff(dst, true);
	WriteFlip(dst, false);
	WriteColors555(dst, base, base);

	uchar best_tbl_idx = 0;
	uchar best_mod_idx = 0;
	uint best_mod_err = UINT_MAX;

	
	
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		
		
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			union Color* color = makeColor(base, lum);

			uint mod_err = getColorError(*src, *color);
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


ulong compressBlock(__global uchar* dst,
										union Color* ver_src,
										union Color* hor_src,
										ulong threshold) {

	unsigned long solid_error = 0;
	if (tryCompressSolidBlock(dst, ver_src, &solid_error)) {
		return solid_error;
	}

	uchar g_idx_to_num[4][8] = {
		{0, 4, 1, 5, 2, 6, 3, 7},        
		{8, 12, 9, 13, 10, 14, 11, 15},  
		{0, 4, 8, 12, 1, 5, 9, 13},      
		{2, 6, 10, 14, 3, 7, 11, 15}     
	};

	union Color* sub_block_src[4] = {ver_src, ver_src + 8, hor_src, hor_src + 8};

	union Color sub_block_avg[4];
	bool use_differential[2] = {true, true};

	
	
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
	for (uint i = 0; i < 4; ++i) {
		for (uint j = 0; j < 8; ++j) {
			sub_block_err[i] += getColorError(sub_block_avg[i], sub_block_src[i][j]);
		}
	}

	bool flip =
	sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1];

	
	memset(dst, 0, 8);

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

	return 0;
}

inline void memcpy_colors(union Color* blocks, int index,
			int offset, __global uchar *src)
{
		uchar *values1;
		uchar *values2;

		values1 = (uchar *) &blocks[index];
		values2 = (uchar *) &blocks[index + 1];

		for (int i = 0; i < 4; i++) {
				values1[i] = *(src+ offset + i);
				values2[i] = *(src+ offset + 4 + i);
		}

}


__kernel void
compression_kernel(__global uchar* src,
		__global uchar* dst,
		int width,
    int height)
{
	int gid_0 = get_global_id(0);
	int gid_1 = get_global_id(1);

	union Color ver_blocks[16];
	union Color hor_blocks[16];

	ulong compressed_error = 0;

	int src_offset = 4 * 4 * gid_0 + gid_1 * width * 4 * 4;
	int dst_offset = gid_0 * 8 + 8 * gid_1 * width / 4;

	src += src_offset;

	for (int x = 0; x < width; x += 4) {
		memcpy_colors(ver_blocks, 0, 0, src + x);
	 	memcpy_colors(ver_blocks, 2, width, src + x);
		memcpy_colors(ver_blocks, 4, width * 2, src + x);
		memcpy_colors(ver_blocks, 6, width * 3, src + x);

		memcpy_colors(hor_blocks, 0, 0, src + x);
		memcpy_colors(hor_blocks, 2, 0, src + x + 8);
		memcpy_colors(hor_blocks, 4, width, src + x);
		memcpy_colors(hor_blocks, 6, width, src + x + 8);

		memcpy_colors(ver_blocks, 8, 0, src + x + 8);
	 	memcpy_colors(ver_blocks, 10, width, src + x + 8);
		memcpy_colors(ver_blocks, 12, width * 2, src + x + 8);
		memcpy_colors(ver_blocks, 14, width * 3, src + x + 8);

		memcpy_colors(hor_blocks, 8, 2 * width, src + x);
	 	memcpy_colors(hor_blocks, 10, 2 * width, src + x + 8);
		memcpy_colors(hor_blocks, 12, 3 * width, src + x);
		memcpy_colors(hor_blocks, 14, 3 * width, src + x + 8);


}
	dst += dst_offset;
	compressed_error += compressBlock(dst, ver_blocks, hor_blocks, UINT_MAX);

}
#include "compress.hpp"


using namespace std;


#define BUF_2M		(2 * 1024 * 1024)
#define BUF_32M		(32 * 1024 * 1024)


#define BUF_128	(128)


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

	
	CL_ERR( clGetPlatformIDs(0, NULL, &platform_num));
	platform_list = new cl_platform_id[platform_num];
	DIE(platform_list == NULL, "alloc platform_list");

	
	CL_ERR( clGetPlatformIDs(platform_num, platform_list, NULL));
	cout << "Platforms found: " << platform_num << endl;

	
	for(uint platf=0; platf<platform_num; platf++)
	{
		
		CL_ERR( clGetPlatformInfo(platform_list[platf],
				CL_PLATFORM_VENDOR, 0, NULL, &attr_size));
		attr_data = new cl_char[attr_size];
		DIE(attr_data == NULL, "alloc attr_data");

		
		CL_ERR( clGetPlatformInfo(platform_list[platf],
				CL_PLATFORM_VENDOR, attr_size, attr_data, NULL));
		cout << "Platform " << platf << " " << attr_data << " ";
		delete[] attr_data;

		
		CL_ERR( clGetPlatformInfo(platform_list[platf],
				CL_PLATFORM_VERSION, 0, NULL, &attr_size));
		attr_data = new cl_char[attr_size];
		DIE(attr_data == NULL, "alloc attr_data");

		
		CL_ERR( clGetPlatformInfo(platform_list[platf],
				CL_PLATFORM_VERSION, attr_size, attr_data, NULL));
		cout << attr_data << endl;
		delete[] attr_data;

		
		platform = platform_list[platf];
		DIE(platform == 0, "platform selection");

		
		if(clGetDeviceIDs(platform,
			CL_DEVICE_TYPE_GPU, 0, NULL, &device_num) == CL_DEVICE_NOT_FOUND) {
			device_num = 0;
			continue;
		}

		device_list = new cl_device_id[device_num];
		DIE(device_list == NULL, "alloc devices");

		
		CL_ERR( clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU,
			  device_num, device_list, NULL));
		cout << "\tDevices found " << device_num  << endl;

		device = device_list[0];

		
		for(uint dev=0; dev<device_num; dev++)
		{
			
			CL_ERR( clGetDeviceInfo(device_list[dev], CL_DEVICE_NAME,
				0, NULL, &attr_size));
			attr_data = new cl_char[attr_size];
			DIE(attr_data == NULL, "alloc attr_data");

			
			CL_ERR( clGetDeviceInfo(device_list[dev], CL_DEVICE_NAME,
				attr_size, attr_data, NULL));
			cout << "\tDevice " << dev << " " << attr_data << " ";
			delete[] attr_data;

			
			CL_ERR( clGetDeviceInfo(device_list[dev], CL_DEVICE_VERSION,
				0, NULL, &attr_size));
			attr_data = new cl_char[attr_size];
			DIE(attr_data == NULL, "alloc attr_data");

			
			CL_ERR( clGetDeviceInfo(device_list[dev], CL_DEVICE_VERSION,
				attr_size, attr_data, NULL));
			cout << attr_data;
			delete[] attr_data;

			
			if((platf == platform_select) && (dev == device_select)){
				device = device_list[dev];
				cout << " <--- SELECTED ";
				break;
			}

			cout << endl;
		}
	}

	delete[] platform_list;
	delete[] device_list;
}


TextureCompressor::TextureCompressor() {

	int platform_select = 0;
	int device_select = 0;

	gpu_find(device, platform_select, device_select);
	DIE(device == 0, "check valid device");

}

TextureCompressor::~TextureCompressor() { }	


unsigned long TextureCompressor::compress(const uint8_t* src,
									  uint8_t* dst,
									  int width,
									  int height)
{

  cl_int ret;
  string kernel_src;

  
  context = clCreateContext(0, 1, &device, NULL, NULL, &ret);
  CL_ERR( ret );

  
  command_queue = clCreateCommandQueue(context, device,
									CL_QUEUE_PROFILING_ENABLE, &ret);
  CL_ERR( ret );

  int source_size = 4 * width * height;
  int destination_size = 4 * width * height / 8;

  
  cl_mem src_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY,
											sizeof(uint8_t) * source_size,
          						NULL, &ret);
  CL_ERR( ret );


  cl_mem dst_buffer = clCreateBuffer(context,	CL_MEM_READ_WRITE,
		 									sizeof(uint8_t) * destination_size,
											NULL, &ret);
  CL_ERR( ret );

  DIE(src_buffer == 0, "alloc src_buffer");
  DIE(dst_buffer == 0, "alloc dst_buffer");

  
  CL_ERR( clEnqueueWriteBuffer(command_queue, src_buffer, CL_TRUE,
						0, sizeof(uint8_t) * source_size, src,
          	0, NULL, NULL));

  
  read_kernel("compression_device.cl", kernel_src);
  const char* kernel_c_str = kernel_src.c_str();

  
  program = clCreateProgramWithSource(context, 1, &kernel_c_str, NULL, &ret);
  CL_ERR( ret );

  
  ret = clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math",
	 			NULL, NULL);
  CL_COMPILE_ERR( ret, program, device );

  
  kernel = clCreateKernel(program, "compression_kernel", &ret);
  CL_ERR( ret );

  
  CL_ERR( clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&src_buffer) );
  CL_ERR( clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&dst_buffer) );
  CL_ERR( clSetKernelArg(kernel, 2, sizeof(cl_uint), (void *)&width) );


  CL_ERR( clSetKernelArg(kernel, 3, sizeof(cl_uint), (void *)&height) );

  
  cl_event event;
  size_t globalSize[2] = {(size_t) width / 4, (size_t) height / 4};
  ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, globalSize, 0, 0,
  			NULL, &event);
  CL_ERR( ret );
  CL_ERR( clWaitForEvents(1, &event));

  
  CL_ERR( clEnqueueReadBuffer(command_queue, dst_buffer, CL_TRUE, 0,
            sizeof(uint8_t) * destination_size, dst, 0, NULL, NULL));

  
  CL_ERR( clFinish(command_queue) );

  
  CL_ERR( clReleaseMemObject(src_buffer) );
  CL_ERR( clReleaseMemObject(dst_buffer) );

  return 0;
}
