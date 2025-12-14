
#define ALIGNAS(X)	__attribute__((aligned(X)))

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


void my_memcpy(union Color *dest, __global union Color *src, int len)

{
    for (int i = 0; i < len; i++)
        dest[i] = src[i];

}


void my_memcpy2(uchar *dest, uchar *src, int len)

{
    for (int i = 0; i < len; i++)
        dest[i] = src[i];

}


void my_memset(__global uchar *dest, uchar val, int len) {
    for (int i = 0; i < len; i++) {
        dest[i] = val;
    }
}


uchar clamp3(int val, int min, int max) {
    return val  max ? max : val);
}




uchar clamp2(uchar val, uchar min, uchar max) {
    return val  max ? max : val);
}

uchar round_to_5_bits(float val) {
    return (uchar)clamp2(val * 31.0f / 255.0f + 0.5f, 0, 31);
}

uchar round_to_4_bits(float val) {
    return (uchar)clamp2(val * 15.0f / 255.0f + 0.5f, 0, 15);
}



ALIGNAS(16) __constant short g_codeword_tables[8][4] = {
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
{0, 4, 1, 5, 2, 6, 3, 7},        
{8, 12, 9, 13, 10, 14, 11, 15},  


{0, 4, 8, 12, 1, 5, 9, 13},      
{2, 6, 10, 14, 3, 7, 11, 15}     
};


union Color makeColor(union Color *base, short lum) {
	int b = (int)(base->channels.b) + lum;
	int g = (int)(base->channels.g) + lum;
	int r = (int)(base->channels.r) + lum;
	union Color color;
	color.channels.b = (uchar)(clamp3(b, 0, 255));
	color.channels.g = (uchar)(clamp3(g, 0, 255));
	color.channels.r = (uchar)(clamp3(r, 0, 255));
	return color;
}



uint getColorError(union Color *u, union Color *v) {
#ifdef USE_PERCEIVED_ERROR_METRIC
	float delta_b = (float)(u->channels.b) - v->channels.b;
	float delta_g = (float)(u->channels.g) - v->channels.g;
	float delta_r = (float)(u->channels.r) - v->channels.r;
	return (uint)(0.299f * delta_b * delta_b +
								 0.587f * delta_g * delta_g +
								 0.114f * delta_r * delta_r);
#else
	int delta_b = (int)(u->channels.b) - v->channels.b;
	int delta_g = (int)(u->channels.g) - v->channels.g;
	int delta_r = (int)(u->channels.r) - v->channels.r;
	return delta_b * delta_b + delta_g * delta_g + delta_r * delta_r;
#endif
}

void WriteColors444(__global uchar* block, const union Color *color0, const union Color *color1) {

	block[0] = (color0->channels.r & 0xf0) | (color1->channels.r >> 4);
	block[1] = (color0->channels.g & 0xf0) | (color1->channels.g >> 4);
	block[2] = (color0->channels.b & 0xf0) | (color1->channels.b >> 4);
}

void WriteColors555(__global uchar* block, const union Color *color0, const union Color *color1) {

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

	short delta_r = (short)(color1->channels.r >> 3) - (color0->channels.r >> 3);
	short delta_g = (short)(color1->channels.g >> 3) - (color0->channels.g >> 3);
	short delta_b = (short)(color1->channels.b >> 3) - (color0->channels.b >> 3);

	
	block[0] = (color0->channels.r & 0xf8) | two_compl_trans_table[delta_r + 4];
	block[1] = (color0->channels.g & 0xf8) | two_compl_trans_table[delta_g + 4];
	block[2] = (color0->channels.b & 0xf8) | two_compl_trans_table[delta_b + 4];
}

void WriteCodewordTable(__global uchar* block, uchar sub_block_id, uchar table) {
    uchar shift = (2 + (3 - sub_block_id * 3));


    block[3] &= ~(0x07 << shift);
    block[3] |= table << shift;
}

void WritePixelData(__global uchar* block, int pixel_data) {
    block[4] |= pixel_data >> 24;
    block[5] |= (pixel_data >> 16) & 0xff;
    block[6] |= (pixel_data >> 8) & 0xff;
    block[7] |= pixel_data & 0xff;
}

void WriteFlip(__global uchar* block, int flip) {
    block[3] &= ~0x01;
    block[3] |= (uchar)(flip);
}

void WriteDiff(__global uchar* block, int diff) {
    block[3] &= ~0x02;
    block[3] |= (uchar)(diff) << 1;
}

void ExtractBlock(uchar* dst, uchar* src, int width) {
	for (int j = 0; j < 4; ++j) {
		my_memcpy2(&dst[j * 4 * 4], src, 4 * 4);
		src += width * 4;
	}
}





union Color makeColor444(float* bgr) {
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





union Color makeColor555(float* bgr) {
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

	float kInv8 = 1.0 / 8.0;
	avg_color[0] = (float)(sum_b) * kInv8;
	avg_color[1] = (float)(sum_g) * kInv8;
	avg_color[2] = (float)(sum_r) * kInv8;
}

ulong computeLuminance(__global uchar* block, union Color* src, union Color* base,
	int sub_block_id, __constant uchar* idx_to_num_tab, ulong threshold)
{
	uint best_tbl_err = threshold;
	uchar best_tbl_idx = 0;
	uchar best_mod_idx[8][8];  

	
	
	for (uint tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		
		
		union Color candidate_color[4];  
		for (uint mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			candidate_color[mod_idx] = makeColor(base, lum);
		}

		uint tbl_err = 0;

		for (uint i = 0; i < 8; ++i) {
			
			
			uint best_mod_err = threshold;
			for (uint mod_idx = 0; mod_idx < 4; ++mod_idx) {
				union Color color = candidate_color[mod_idx];

				uint mod_err = getColorError(&src[i], &color);
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


int tryCompressSolidBlock(__global uchar* dst, union Color* src, ulong* error)
{


	for (uint i = 1; i < 16; ++i) {
		if (src[i].bits != src[0].bits)
			return 0;
	}

	
	my_memset(dst, 0, 8);



	float src_color_float[3] = {(float)(src->channels.b),
	                            (float)(src->channels.g),
	                            (float)(src->channels.r)};
	union Color base = makeColor555(src_color_float);

	WriteDiff(dst, 1);
	WriteFlip(dst, 0);
	WriteColors555(dst, &base, &base);

	uchar best_tbl_idx = 0;
	uchar best_mod_idx = 0;
	uint best_mod_err = 4294967295;

	
	
	for (uint tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		
		
		for (uint mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			union Color color = makeColor(&base, lum);

			uint mod_err = getColorError(src, &color);
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
	for (uint i = 0; i < 2; ++i) {
		for (uint j = 0; j < 8; ++j) {
		
			int texel_num = g_idx_to_num[i][j];


			pix_data |= msb << (texel_num + 16);
			pix_data |= lsb << (texel_num);
		}
	}

	WritePixelData(dst, pix_data);
	*error = 16 * best_mod_err;
	return 1;
}

ulong compressBlock(__global uchar* dst, union Color* ver_src,
	union Color* hor_src, ulong threshold)
{
	ulong solid_error = 0;
	if (tryCompressSolidBlock(dst, ver_src, &solid_error)) {
		return solid_error;
	}

	union Color* sub_block_src[4] = {ver_src, ver_src + 8, hor_src, hor_src + 8};

	union Color sub_block_avg[4];
	int use_differential[2] = {1, 1};

	
	
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
				use_differential[i / 2] = 0;
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
			sub_block_err[i] += getColorError(&sub_block_avg[i], &(sub_block_src[i][j]));
		}
	}

	int flip = 0;
	if (sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1])
		flip = 1;

	
	my_memset(dst, 0, 8);

	WriteDiff(dst, use_differential[!!flip]);
	WriteFlip(dst, flip);

	uchar sub_block_off_0 = flip ? 2 : 0;
	uchar sub_block_off_1 = sub_block_off_0 + 1;

	if (use_differential[!!flip]) {
		WriteColors555(dst, &sub_block_avg[sub_block_off_0],
	        &sub_block_avg[sub_block_off_1]);
	} else {
		WriteColors444(dst, &sub_block_avg[sub_block_off_0],
	        &sub_block_avg[sub_block_off_1]);
	}

	ulong lumi_error1 = 0, lumi_error2 = 0;

	
	lumi_error1 = computeLuminance(dst, sub_block_src[sub_block_off_0],
	                               &sub_block_avg[sub_block_off_0], 0,
	                               g_idx_to_num[sub_block_off_0],
	                               threshold);
	
	lumi_error2 = computeLuminance(dst, sub_block_src[sub_block_off_1],
	                               &sub_block_avg[sub_block_off_1], 1,
	                               g_idx_to_num[sub_block_off_1],
	                               threshold);

	return lumi_error1 + lumi_error2;


}

__kernel void mat_mul(__global uchar* src, __global uchar* dst, int height, int width) {
	
	int gid_0 = get_global_id(0);
	int gid_1 = get_global_id(1);

	
	int y = gid_0 * 4;
	int x = gid_1 * 4;
	
	
	dst += 8 * gid_0 * width / 4 + 8 * gid_1;
	src += gid_0 * 4 * 4 * width;


	union Color ver_blocks[16];
	union Color hor_blocks[16];

	ulong compressed_error = 0;
	
	
	__global union Color* row0 = (__global union Color*)(src + x * 4);
	__global union Color* row1 = row0 + width;
	__global union Color* row2 = row1 + width;
	__global union Color* row3 = row2 + width;

	
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

	
	compressBlock(dst, ver_blocks, hor_blocks, 4294967295);

}
#include "compress.hpp"
#include 
#include 
#include 
#include 
#include 
#include 
#include 

using namespace std;

void read_kernel(string file_name, string &str_kernel)
{
	ifstream in_file(file_name.c_str());
	in_file.open(file_name.c_str());
	
	stringstream str_stream;
	str_stream << in_file.rdbuf();

	str_kernel = str_stream.str();
}

unsigned long gpu_profile_kernel(cl_device_id device, const uint8_t *src, uint8_t *dst,
				int width, int height)
{
	
	cl_int ret;
	cl_context context;
	cl_command_queue cmdQueue;
	cl_program program;
	cl_kernel kernel;
	string kernel_src;

	
	int bufMatSizeA = width * height * 4;
	int bufMatSizeC = width * height * 4 / 8;

	
	int width2 = width / 4;
	int height2 = height / 4;

	
	context = clCreateContext(0, 1, &device, NULL, NULL, &ret);
	
	
	cmdQueue = clCreateCommandQueue(context, device, 0, &ret);
	
	
	
	cl_mem bufMatA = clCreateBuffer(context, CL_MEM_READ_ONLY,
		bufMatSizeA, NULL, &ret);
	
	
	cl_mem bufMatC = clCreateBuffer(context, CL_MEM_READ_WRITE,
		bufMatSizeC, NULL, &ret);
		
	
	clEnqueueWriteBuffer(cmdQueue, bufMatA, CL_TRUE, 0,
		bufMatSizeA, src, 0, NULL, NULL);

	
	read_kernel("dima.cl", kernel_src);
	const char* kernel_c_str = kernel_src.c_str();

	
	program = clCreateProgramWithSource(context, 1,
		(const char **) &kernel_c_str, NULL, &ret);

	
	ret = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

	
	size_t log_size;
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    
    	char *log = (char *) malloc(log_size);
  
    	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
	

	
	kernel = clCreateKernel(program, "mat_mul", &ret);

	
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufMatA);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufMatC);
	clSetKernelArg(kernel, 2, sizeof(int), &height);
	clSetKernelArg(kernel, 3, sizeof(int), &width);

	
	size_t globalSize[2] = {(size_t) height2, (size_t) width2};
	ret = clEnqueueNDRangeKernel(cmdQueue, kernel, 2, NULL,
		globalSize, 0, 0, NULL, NULL);

	clFinish(cmdQueue);

	
	clEnqueueReadBuffer(cmdQueue, bufMatC, CL_TRUE, 0,
		bufMatSizeC, dst, 0, NULL, NULL);

		
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseMemObject(bufMatA);
	clReleaseMemObject(bufMatC);
	clReleaseCommandQueue(cmdQueue);
	clReleaseContext(context);
	
	return 0;

}





unsigned long TextureCompressor::compress(const uint8_t* src,
					  uint8_t* dst,
					  int width,
					  int height)
{
	return gpu_profile_kernel(device, src, dst, width, height);
}


void gpu_find(cl_device_id &device)
{
	
	cl_char contor = 0;
	cl_platform_id platform;
	cl_uint platform_num = 0;
	cl_platform_id* platform_list = NULL;

	cl_uint device_num = 0;
	cl_device_id* device_list = NULL;

	size_t attr_size = 0;
	cl_char* attr_data = NULL;

	
	clGetPlatformIDs(0, NULL, &platform_num);
	platform_list = new cl_platform_id[platform_num];

	
	clGetPlatformIDs(platform_num, platform_list, NULL);
	cout << "Platforms found!: " << platform_num << endl;

	
	for(uint platf=0; platf<platform_num; platf++)
	{
		
		clGetPlatformInfo(platform_list[platf],
				CL_PLATFORM_VENDOR, 0, NULL, &attr_size);
		attr_data = new cl_char[attr_size];

		
		clGetPlatformInfo(platform_list[platf],
				CL_PLATFORM_VENDOR, attr_size, attr_data, NULL);
		cout << "Platform " << platf << " " << attr_data;
		delete[] attr_data;

		
		clGetPlatformInfo(platform_list[platf],
				CL_PLATFORM_VERSION, 0, NULL, &attr_size);
		attr_data = new cl_char[attr_size];

		
		clGetPlatformInfo(platform_list[platf],
				CL_PLATFORM_VERSION, attr_size, attr_data, NULL);
		cout << attr_data << endl;
		delete[] attr_data;

		
		platform = platform_list[platf];

		
		clGetDeviceIDs(platform, 
			CL_DEVICE_TYPE_ALL, 0, NULL, &device_num);

		device_list = new cl_device_id[device_num];

		
		clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL,
			device_num, device_list, NULL);
		cout << "\tDevices found " << device_num  << endl;

		
		for(uint dev=0; dev<device_num; dev++)
		{
			
			clGetDeviceInfo(device_list[dev], CL_DEVICE_NAME,
				0, NULL, &attr_size);
			attr_data = new cl_char[attr_size];

			
			clGetDeviceInfo(device_list[dev], CL_DEVICE_NAME,
				attr_size, attr_data, NULL);
			cout << "\tDevice " << dev << " " << attr_data << " ";
			delete[] attr_data;

				
			cl_char* aux = new cl_char[attr_size];
			clGetDeviceInfo(device_list[dev], CL_DEVICE_NAME,
				attr_size, aux, NULL);

			
			clGetDeviceInfo(device_list[dev], CL_DEVICE_VERSION,
				0, NULL, &attr_size);
			attr_data = new cl_char[attr_size];
			
			
			clGetDeviceInfo(device_list[dev], CL_DEVICE_VERSION,
				attr_size, attr_data, NULL);
			cout << attr_data; 
			delete[] attr_data;

			
			if(strstr((char*)aux, "Tesla") != NULL && contor == 0){
				contor = 1;
				device = device_list[dev];
				cout << " <--- SELECTED ";
			}
			delete[] aux;

			cout << endl;
		}
	}

	delete[] platform_list;
	delete[] device_list;
}


TextureCompressor::TextureCompressor() {
	gpu_find(device);
}

TextureCompressor::~TextureCompressor() { }	
	
