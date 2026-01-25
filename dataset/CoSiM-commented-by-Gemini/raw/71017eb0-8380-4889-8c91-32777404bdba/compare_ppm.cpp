
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
>>>> file: compress.cl
#define ALIGNAS(X)	__attribute__((aligned(X)))
#define ALIGNMENT 16 

typedef uchar uint8_t;
typedef short int16_t;
typedef uint uint32_t;

typedef union Color {
	struct BgraColorType {
		uint8_t b;
		uint8_t g;
		uint8_t r;
		uint8_t a;
	} channels;
	uint8_t components[4];
	uint32_t bits;
}Color;

void mmemcpy(void *bdst,void *bsrc, int len)
{
	char *dst;
	char *src;
	dst = (char*)(bdst);
	src = (char*)(bsrc);
	for(int i = 0 ; i < len ; i++)
	{
		dst[i] = src[i];
	}
}

void mmemset(void *ptr,int value,int len)
{
	char *p;
	p = (char*)(ptr);
	for(int i = 0 ; i < len ; i ++)
	{
		p[i] = value;
	}
}

uchar round_to_5_bits(float value)
{
	float temp;
	float _min_;
	float _max_; 
	_min_ = 0;
	_max_ = 31;
	temp = (value * 31 / 255) + 0.5;
	return (uchar)(clamp(temp,_min_,_max_));;
}

uchar round_to_4_bits(float value)
{
	__local float temp;
	__local float _min_;
	__local float _max_; 
	_min_ = 0;
	_max_ = 15;
	temp = (value * 15 / 255) + 0.5;
	return (uchar)(clamp(temp,_min_,_max_));;
}

__constant int16_t g_codeword_tables[8][4] = {
	{-8, -2, 2, 8},
	{-17, -5, 5, 17},
	{-29, -9, 9, 29},
	{-42, -13, 13, 42},
	{-60, -18, 18, 60},
	{-80, -24, 24, 80},
	{-106, -33, 33, 106},
	{-183, -47, 47, 183}};

__constant uint8_t g_mod_to_pix[4] = {3, 2, 0, 1};
__constant uint8_t g_idx_to_num[4][8] = {
	{0, 4, 1, 5, 2, 6, 3, 7},        
	{8, 12, 9, 13, 10, 14, 11, 15},  
	{0, 4, 8, 12, 1, 5, 9, 13},      
	{2, 6, 10, 14, 3, 7, 11, 15}     
};

__constant uint8_t two_compl_trans_table[8] = {
		4,  // -4 (100b)
		5,  // -3 (101b)
		6,  // -2 (110b)
		7,  // -1 (111b)
		0,  //  0 (000b)
		1,  //  1 (001b)
		2,  //  2 (010b)
		3,  //  3 (011b)
	};

Color makeColor(Color *base, int16_t lum) {
	int b;
	int g;
	int r;
	b = (int)(base->channels.b) + lum;
	g = (int)(base->channels.g) + lum;
	r = (int)(base->channels.r) + lum;
	Color color;
	color.channels.b = (uchar)(clamp(b, 0, 255));
	color.channels.g = (uchar)(clamp(g, 0, 255));
	color.channels.r = (uchar)(clamp(r, 0, 255));
	return color;
}

uint32_t getColorError(Color *u, Color *v) {
#ifdef USE_PERCEIVED_ERROR_METRIC
	float delta_b;
	float delta_g;
	float delta_r;
	delta_b = (float)(u->channels.b) - v->channels.b;
	delta_g = (float)(u->channels.g) - v->channels.g;
	delta_r = (float)(u->channels.r) - v->channels.r;
	return convert_uint(0.299 * delta_b * delta_b + 0.587 * delta_g * delta_g + 0.114 * delta_r * delta_r);
#else
	int delta_b;
	int delta_g;
	int delta_r;
	delta_b = (int)(u->channels.b) - v->channels.b;
	delta_g = (int)(u->channels.g) - v->channels.g;
	delta_r = (int)(u->channels.r) - v->channels.r;
	return delta_b * delta_b + delta_g * delta_g + delta_r * delta_r;
#endif
}

void WriteColors444(uint8_t *block, Color *color0, Color *color1) {

	block[0] = (color0->channels.r & 0xf0) | (color1->channels.r >> 4);
	block[1] = (color0->channels.g & 0xf0) | (color1->channels.g >> 4);
	block[2] = (color0->channels.b & 0xf0) | (color1->channels.b >> 4);
}

void WriteColors555(uint8_t* block, Color *color0, Color *color1) {
	
	int16_t delta_r;
	int16_t delta_g;
	int16_t delta_b;
	delta_r = (short)(color1->channels.r >> 3) - (color0->channels.r >> 3);
	delta_g = (short)(color1->channels.g >> 3) - (color0->channels.g >> 3);
	delta_b = (short)(color1->channels.b >> 3) - (color0->channels.b >> 3);
	
	block[0] = (color0->channels.r & 0xf8) | two_compl_trans_table[delta_r + 4];
	block[1] = (color0->channels.g & 0xf8) | two_compl_trans_table[delta_g + 4];
	block[2] = (color0->channels.b & 0xf8) | two_compl_trans_table[delta_b + 4];

}

void WriteCodewordTable(uint8_t *block, uint8_t sub_block_id, uint8_t table) {
	
	uint8_t shift; 
	shift = (2 + (3 - sub_block_id * 3));
	block[3] &= ~(0x07 << shift);
	block[3] |= table << shift;
}

void WritePixelData(uint8_t *block, uint32_t pixel_data) {
	block[4] |= pixel_data >> 24;
	block[5] |= (pixel_data >> 16) & 0xff;
	block[6] |= (pixel_data >> 8) & 0xff;
	block[7] |= pixel_data & 0xff;
}

void WriteFlip(uint8_t *block, bool flip) {
	block[3] &= ~0x01;
	block[3] |= flip;
}



void WriteDiff(uint8_t* block, bool diff) {
	block[3] &= ~0x02;
	block[3] |= diff << 1;
}

void ExtractBlock(uint8_t *dst, uint8_t *src, uint width) {
	int j;
	for(j = 0; j < 4;++j) {
		mmemcpy((void*)(&dst[j*16]),(void*)src,4*4);
		src += width * 4;
	}
}

Color makeColor444(float *bgr) {
	uint8_t b4;
	uint8_t g4;
	uint8_t r4;
	b4 = round_to_4_bits(bgr[0]);
	g4 = round_to_4_bits(bgr[1]);
	r4 = round_to_4_bits(bgr[2]);
	Color bgr444;
	bgr444.channels.b = (b4 << 4) | b4;
	bgr444.channels.g = (g4 << 4) | g4;
	bgr444.channels.r = (r4 << 4) | r4;
	bgr444.channels.a = 0x44;
	return bgr444;
}

Color makeColor555(float *bgr) {

	uint8_t b5;
	uint8_t g5;
	uint8_t r5;
	b5 = round_to_5_bits(bgr[0]);
	g5 = round_to_5_bits(bgr[1]);
	r5 = round_to_5_bits(bgr[2]);
	Color bgr555;
	bgr555.channels.b = (b5 > 2);
	bgr555.channels.g = (g5 > 2);
	bgr555.channels.r = (r5 > 2);
	bgr555.channels.a = 0x55;
	return bgr555;
}

__constant float kInv8 = 0.125;

void getAverageColor(Color *src, float *avg_color)
{
	uint32_t sum_b;
	uint32_t sum_g;
	uint32_t sum_r;
	uint i;
	
	sum_b = 0;
	sum_g = 0;
	sum_r = 0;



	for (i = 0; i < 8; ++i) {
		sum_b += src[i].channels.b;
		sum_g += src[i].channels.g;
		sum_r += src[i].channels.r;
	}

	avg_color[0] = convert_float(sum_b) * kInv8;
	avg_color[1] = convert_float(sum_g) * kInv8;
	avg_color[2] = convert_float(sum_r) * kInv8;
}

ulong computeLuminance(uint8_t *block, Color* src,Color *base,int sub_block_id,__constant uint8_t *idx_to_num_tab,ulong threshold)
{
	uint32_t best_tbl_err;
	uint8_t best_tbl_idx;
	best_tbl_err = threshold;
	best_tbl_idx = 0;
	uint8_t best_mod_idx[8][8];  // [table][texel]
	uint tbl_idx;
	uint mod_idx;
	uint i;
	uint32_t tbl_err;
	uint32_t best_mod_err;	
	uint pix_idx;
	uint32_t mod_err;

	for (tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		Color candidate_color[4];  // [modifier]
		for (mod_idx = 0; mod_idx < 4; ++mod_idx) {
			int16_t lum = g_codeword_tables[tbl_idx][mod_idx];
			candidate_color[mod_idx] = makeColor(base, lum);
		}
		
	tbl_err = 0;
	for (i = 0; i < 8; ++i) {
		best_mod_err = threshold;
		for (mod_idx = 0; mod_idx < 4; ++mod_idx) {
			
			const Color *color; 
			color = &(candidate_color[mod_idx]);


			mod_err = getColorError(&(src[i]), color);
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
	pix_data = 0;
	for (i = 0; i < 8; ++i) {
		mod_idx = best_mod_idx[best_tbl_idx][i];
		pix_idx = g_mod_to_pix[mod_idx];
		
		uint32_t lsb; 
		lsb = pix_idx & 0x1;
		uint32_t msb;
		msb = pix_idx >> 1;
		
		// Obtain the texel number as specified in the standard.
		int texel_num;
		texel_num = idx_to_num_tab[i];
		pix_data |= msb << (texel_num + 16);
		pix_data |= lsb << (texel_num);
	}

	WritePixelData(block, pix_data);
	return best_tbl_err;
}

bool tryCompressSolidBlock(uint8_t *dst, Color *src, ulong *error)
{
	uint i;
	for (i = 1; i < 16; ++i) {
		if (src[i].bits != src[0].bits)
			return false;
	}
	
	// Clear destination buffer so that we can "or" in the results.
	mmemset(dst, 0, 8);
	 
	
	float src_color_float[3];
	src_color_float[0] = (float)(src->channels.b);
	src_color_float[1] = (float)(src->channels.g);
	src_color_float[2] = (float)(src->channels.r);
	Color base;
	base = makeColor555(src_color_float);
	
	WriteDiff(dst, true);
	WriteFlip(dst, false);
	WriteColors555(dst, &base, &base);
	
	uint8_t best_tbl_idx;
	best_tbl_idx = 0;
	uint8_t best_mod_idx;
	best_mod_idx = 0;
	uint32_t best_mod_err;
	best_mod_err = UINT_MAX; 
	
	// Try all codeword tables to find the one giving the best results for this
	// block.
	uint tbl_idx;
	uint mod_idx;
	for (tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		// Try all modifiers in the current table to find which one gives the
		// smallest error.
		for (mod_idx = 0; mod_idx < 4; ++mod_idx) {
			int16_t lum;
			lum = g_codeword_tables[tbl_idx][mod_idx];
			const Color color = makeColor(&base, lum);
			
			uint32_t mod_err;
			mod_err = getColorError(src, &color);
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
	
	uint8_t pix_idx;
	pix_idx = g_mod_to_pix[best_mod_idx];
	uint32_t lsb;
	lsb = pix_idx & 0x1;
	uint32_t msb;
	msb = pix_idx >> 1;
	
	uint32_t pix_data;
	pix_data = 0;
	uint j;
	for (i = 0; i < 2; ++i) {
		for (j = 0; j < 8; ++j) {
			// Obtain the texel number as specified in the standard.
			int texel_num;
			texel_num = g_idx_to_num[i][j];
			pix_data |= msb << (texel_num + 16);
			pix_data |= lsb << (texel_num);
		}
	}
	
	WritePixelData(dst, pix_data);
	*error = 16 * best_mod_err;
	return true;
}

ulong compressBlock(uint8_t *dst,Color *ver_src, Color *hor_src, ulong threshold)
{
	ulong solid_error;
	solid_error = 0;

	if (tryCompressSolidBlock(dst, ver_src, &solid_error)) {
		return solid_error;
	}
	
	const Color* sub_block_src[4] = {ver_src, ver_src + 8, hor_src, hor_src + 8};
	
	Color sub_block_avg[4];
	bool use_differential[2];
	use_differential[0] = true;
	use_differential[1] = true;
	
	// Compute the average color for each sub block and determine if differential
	// coding can be used.
	uint i;
	uint j;

	for (i = 0, j = 1; i < 4; i += 2, j += 2) {
		float avg_color_0[3];
		getAverageColor(sub_block_src[i], avg_color_0);
		Color avg_color_555_0 = makeColor555(avg_color_0);
		
		float avg_color_1[3];
		getAverageColor(sub_block_src[j], avg_color_1);
		Color avg_color_555_1 = makeColor555(avg_color_1);
		uint light_idx;
		for (light_idx = 0; light_idx < 3; ++light_idx) {
			int u;
			int v;
			u = avg_color_555_0.components[light_idx] >> 3;
			v = avg_color_555_1.components[light_idx] >> 3;
			
			int component_diff;
			component_diff = v - u;
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
	uint32_t sub_block_err[4];
	sub_block_err[0] = 0;
	sub_block_err[1] = 0;
	sub_block_err[2] = 0;
	sub_block_err[3] = 0;
	for (i = 0; i < 4; ++i) {


		for (j = 0; j < 8; ++j) {
			sub_block_err[i] += getColorError(&(sub_block_avg[i]), &(sub_block_src[i][j]));
		}
	}
	
	bool flip = sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1];
	
	// Clear destination buffer so that we can "or" in the results.
	mmemset(dst, 0, 8);
 
	WriteDiff(dst, use_differential[!!flip]);
	WriteFlip(dst, flip);
	
	uint8_t sub_block_off_0 = flip ? 2 : 0;
	uint8_t sub_block_off_1 = sub_block_off_0 + 1;
	
	if (use_differential[!!flip]) {
		WriteColors555(dst, &(sub_block_avg[sub_block_off_0]),
					   &(sub_block_avg[sub_block_off_1]));
	} else {
		WriteColors444(dst, &(sub_block_avg[sub_block_off_0]),
					   &(sub_block_avg[sub_block_off_1]));
	}
	
	ulong lumi_error1;
	ulong lumi_error2;
	lumi_error1 = 0;
	lumi_error2 = 0;
	



	lumi_error1 = computeLuminance(dst, sub_block_src[sub_block_off_0],
								   &(sub_block_avg[sub_block_off_0]), 0,
								   g_idx_to_num[sub_block_off_0],
								   threshold);

	lumi_error2 = computeLuminance(dst, sub_block_src[sub_block_off_1],
								   &(sub_block_avg[sub_block_off_1]), 1,
								   g_idx_to_num[sub_block_off_1],
								   threshold);
	return lumi_error1 + lumi_error2;
}

__kernel void compress(__global uchar *src,__global uchar *dst, __global int *dims)
{
	uint8_t aux[8];
	//uint8_t *aux;
	int x;
	int y;	
 	int soffset;
	int doffset; 
	int width = dims[0];
 	int height = dims[1];
 	Color ver[16];
 	Color hor[16];
  	
	y = get_global_id(0);
 	x = get_global_id(1);

	
	soffset= width * 16 * y + x * 16;
 	doffset = width * 2 * y + x * 8;

	
 	const Color* row0 = src + soffset;
 	const Color* row1 = row0 + width;
 	const Color* row2 = row1 + width;
 	const Color* row3 = row2 + width;
  	 
 	mmemcpy((void *)ver, (void *)row0, 8);
 	mmemcpy((void *)(ver + 2), (void *)row1, 8);
 	mmemcpy((void *)(ver + 4), (void *)row2, 8);
	mmemcpy((void *)(ver + 6), (void *)row3, 8);
 	mmemcpy((void *)(ver + 8), (void *)(row0 + 2), 8);
 	mmemcpy((void *)(ver + 10), (void *)(row1 + 2), 8);
 	mmemcpy((void *)(ver + 12), (void *)(row2 + 2), 8);
 	mmemcpy((void *)(ver + 14), (void *)(row3 + 2), 8);   
 	mmemcpy((void *)hor, (void *)row0, 16);
 	mmemcpy((void *)(hor + 4), (void *)row1, 16);
 	mmemcpy((void *)(hor + 8), (void *)row2, 16);
 	mmemcpy((void *)hor + 12, (void *)row3, 16);
	

 	compressBlock(&(aux[0]), ver, hor, ULONG_MAX);	
	for(int i = 0;i < 8;i++)
 	{
  		dst[doffset + i] = aux[i];
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

#if __APPLE__
#include 
#else
#include 
#endif

#define NOTREACHED()

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
	
protected:
	unsigned long compressBlock(uint8_t* dst,
								const Color* ver_src,
								const Color* hor_src,
								unsigned long threshold);
};

#endif  // CC_RASTER_TEXTURE_COMPRESSOR_H_
>>>> file: texture_compress.cpp
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
#include  
#include 
using namespace std; 

void gpu_find(cl_device_id &device, uint device_select); TextureCompressor::TextureCompressor() {
	gpu_find(this->device, 0);
} 	
TextureCompressor::~TextureCompressor() { }	// destructor

#define DIE(assertion, call_description) \
do { \
	if (assertion) { \
		fprintf(stderr, "(%d): ", __LINE__); \
		perror(call_description); \
		exit(EXIT_FAILURE); \
	} \
} while(0);

const char* cl_get_string_err(cl_int err) { switch (err) {
  case CL_SUCCESS: return "Success!";
  case CL_DEVICE_NOT_FOUND: return "Device not found.";
  case CL_DEVICE_NOT_AVAILABLE: return "Device not available";
  case CL_COMPILER_NOT_AVAILABLE: return "Compiler not available";
  case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "Memory object alloc fail";
  case CL_OUT_OF_RESOURCES: return "Out of resources";
  case CL_OUT_OF_HOST_MEMORY: return "Out of host memory";
  case CL_PROFILING_INFO_NOT_AVAILABLE: return "Profiling information N/A";
  case CL_MEM_COPY_OVERLAP: return "Memory copy overlap";
  case CL_IMAGE_FORMAT_MISMATCH: return "Image format mismatch";
  case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "Image format no support";
  case CL_BUILD_PROGRAM_FAILURE: return "Program build failure";
  case CL_MAP_FAILURE: return "Map failure";
  case CL_INVALID_VALUE: return "Invalid value";
  case CL_INVALID_DEVICE_TYPE: return "Invalid device type";
  case CL_INVALID_PLATFORM: return "Invalid platform";
  case CL_INVALID_DEVICE: return "Invalid device";
  case CL_INVALID_CONTEXT: return "Invalid context";
  case CL_INVALID_QUEUE_PROPERTIES: return "Invalid queue properties";
  case CL_INVALID_COMMAND_QUEUE: return "Invalid command queue";
  case CL_INVALID_HOST_PTR: return "Invalid host pointer";
  case CL_INVALID_MEM_OBJECT: return "Invalid memory object";
  case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:return "Invalid image format desc";
  case CL_INVALID_IMAGE_SIZE: return "Invalid image size";
  case CL_INVALID_SAMPLER: return "Invalid sampler";
  case CL_INVALID_BINARY: return "Invalid binary";
  case CL_INVALID_BUILD_OPTIONS: return "Invalid build options";
  case CL_INVALID_PROGRAM: return "Invalid program";
  case CL_INVALID_PROGRAM_EXECUTABLE: return "Invalid program exec";
  case CL_INVALID_KERNEL_NAME: return "Invalid kernel name";
  case CL_INVALID_KERNEL_DEFINITION: return "Invalid kernel definition";
  case CL_INVALID_KERNEL: return "Invalid kernel";
  case CL_INVALID_ARG_INDEX: return "Invalid argument index";
  case CL_INVALID_ARG_VALUE: return "Invalid argument value";
  case CL_INVALID_ARG_SIZE: return "Invalid argument size";
  case CL_INVALID_KERNEL_ARGS: return "Invalid kernel arguments";
  case CL_INVALID_WORK_DIMENSION: return "Invalid work dimension";
  case CL_INVALID_WORK_GROUP_SIZE: return "Invalid work group size";
  case CL_INVALID_WORK_ITEM_SIZE: return "Invalid work item size";
  case CL_INVALID_GLOBAL_OFFSET: return "Invalid global offset";
  case CL_INVALID_EVENT_WAIT_LIST: return "Invalid event wait list";
  case CL_INVALID_EVENT: return "Invalid event";
  case CL_INVALID_OPERATION: return "Invalid operation";
  case CL_INVALID_GL_OBJECT: return "Invalid OpenGL object";
  case CL_INVALID_BUFFER_SIZE: return "Invalid buffer size";
  case CL_INVALID_MIP_LEVEL: return "Invalid mip-map level";
  default: return "Unknown";
  }
}
void cl_get_compiler_err_log(cl_program program, cl_device_id device) {
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
void read_kernel(string file_name, string &str_kernel) {
	ifstream in_file(file_name.c_str());
	in_file.open(file_name.c_str());
	DIE( !in_file.is_open(), "ERR OpenCL kernel file. Same directory as binary ?" );
	stringstream str_stream;
	str_stream << in_file.rdbuf();
	str_kernel = str_stream.str();
}
int CL_ERR(int cl_ret) {
	if(cl_ret != CL_SUCCESS){
		cout << endl << cl_get_string_err(cl_ret) << endl;
		return 1;
	}
	return 0;
}
int CL_COMPILE_ERR(int cl_ret, cl_program program, cl_device_id device) {
	if(cl_ret != CL_SUCCESS){
		cout << endl << cl_get_string_err(cl_ret) << endl;
		cl_get_compiler_err_log(program, device);
		return 1;
	}
	return 0;
}
void gpu_find(cl_device_id &device, uint device_select) {
	cl_platform_id platform;
	cl_uint platform_num = 0;
	cl_platform_id* platform_list = NULL;
	cl_uint device_num = 0;
	cl_device_id* device_list = NULL;
	int flag = 0;
	/* get num of available OpenCL platforms */
	CL_ERR( clGetPlatformIDs(0, NULL, &platform_num));
	platform_list = new cl_platform_id[platform_num];
	DIE(platform_list == NULL, "alloc platform_list");
	/* get all available OpenCL platforms */
	CL_ERR( clGetPlatformIDs(platform_num, platform_list, NULL));
	/* list all platforms and VENDOR/VERSION properties */
	for(uint platf=0; platf<platform_num; platf++)
	{
	
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
		/* list all devices and TYPE/VERSION properties */
		for(uint dev=0; dev<device_num; dev++)
		{
			/* select device based on cli arguments */
			if(dev == device_select)
			{
				device = device_list[dev];
				flag = 1;
				break;
			}
		}
		if(flag == 1)
		{
			break;
		}
	}
	delete[] platform_list;
	delete[] device_list;
}
void gpu_execute_kernel(cl_device_id device, const uint8_t *src, uint8_t *dst, int width, int height) {
	string kernel_src;
	cl_context context;
	cl_command_queue commands;
	cl_program program;
	cl_kernel kernel;
	cl_uint nd;
	cl_mem source;
	cl_mem destination;
	cl_mem dimensions;
	size_t global[2];
	size_t local[2];
	int i, ret, dim[2];
	int srcSz, dstSz;
	/* creez context pentru device*/
	context = clCreateContext(0, 1, &device, NULL, NULL, &ret);
	CL_ERR(ret);
	commands = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &ret);
	CL_ERR(ret);
	/*setez buffer-e, si le scriu in memoria globala*/
	dim[0] = width;
	dim[1] = height;
	srcSz = width * height * 4;
	dstSz = srcSz / 8;


	dimensions = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * 2, NULL, NULL);
	source = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(uint8_t) * srcSz, NULL, NULL);
	destination = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint8_t) * dstSz, NULL, NULL);
	/*determin sursa kernel*/
	read_kernel("compress.cl", kernel_src);
	const char *kernel_c_str = kernel_src.c_str();
	/*creez programul de executat*/


	program = clCreateProgramWithSource(context, 1, (const char **) &kernel_c_str, NULL, &ret);
	CL_ERR(ret);
	/*build program*/
	ret = clBuildProgram(program, 1,&device, NULL, NULL, NULL);
	CL_COMPILE_ERR(ret, program, device);
	/*creeaza kernel-ul*/
	kernel = clCreateKernel(program, "compress", &ret);
	CL_ERR(ret);
	//setez argumentele pentru kernel
	ret = 0;
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &source);
	ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &destination);


	ret |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &dimensions);
	/*populez zona de memorie pentru sursa*/
	ret = clEnqueueWriteBuffer(commands, source, CL_TRUE, 0, sizeof(uint8_t) * srcSz, src, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(commands, dimensions, CL_TRUE, 0, sizeof(int) *2, dim, 0, NULL, NULL);
	cl_event prof_event;
	//


	global[0] = (size_t)height/4;
	global[1] = (size_t)width/4;
	//local[0] = (size_t)1;
	//local[1] = (size_t)1;
	//for(int i = 0 ; i < 2 ; i++)
	ret = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, global, NULL, 0, NULL, NULL);
	
	/*astept finalizarea calculelor*/
	clFinish(commands);
	/*citesc date*/
	ret = clEnqueueReadBuffer(commands, destination, CL_TRUE, 0, sizeof(uint8_t) * dstSz, dst, 0, NULL, NULL);
	/*clean*/
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseMemObject(source);
	clReleaseMemObject(destination);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);
}
unsigned long TextureCompressor::compress(const uint8_t* src, uint8_t* dst, int width, int height) {
	gpu_execute_kernel(this->device, src, dst, width, height);
	return 0;
}
