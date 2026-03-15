
# define INT32_MAX              (2147483647)
# define UINT32_MAX             (4294967295U)

#define ALIGNAS(X)      __attribute__((aligned(X)))

typedef union Color {
    struct BgraColorType {
        uchar b;
        uchar g;
        uchar r;
        uchar a;
    } channels;
    uchar components[4];
    uint bits;
}Color;



void memcpy(void *dst, __global void * src, int n){
    __global char * csrc= (__global char *) src;
    char * cdst= ( char *) dst;
    for(int i = 0;i < n;i++){
        cdst[i]=csrc[i];
    }

}



void memcpy2(__global void *dst,__global  void * src, int n){
    __global char * csrc= (__global char *) src;
    __global char * cdst= (__global char *)  dst;
    for(int i = 0;i < n;i++){
        cdst[i]=csrc[i];
    }

}



void memset( __global void *dst, int value, int n){
    __global char * cdst= (__global char *) dst;
    for(int i = 0;i < n;i++){
        cdst[i]=value;
    }

}

inline uchar clamp2(uchar val, uchar min, uchar max) {
    return val < min ? min : (val > max ? max : val);
}

inline uchar clamp3(int val, int min, int max) {
    return val < min ? min : (val > max ? max : val);
}


inline uchar round_to_5_bits(float val) {
    return clamp2(val * 31.0f / 255.0f + 0.5f, 0, 31);
}

inline uchar round_to_4_bits(float val) {
    return clamp2(val * 15.0f / 255.0f + 0.5f, 0, 15);
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





inline Color makeColor(const Color *  base, short lum) {
    int b = (int)(base->channels.b) + lum;
    int g = (int)(base->channels.g) + lum;
    int r = (int)(base->channels.r) + lum;
    Color color;

    color.channels.b = (clamp3(b, 0, 255));
    color.channels.g = (clamp3(g, 0, 255));


    color.channels.r = (clamp3(r, 0, 255));
    return color;
}



inline uint getColorError(const Color * u, const Color * v) {
    #ifdef USE_PERCEIVED_ERROR_METRIC
        float delta_b = (float)(u->channels.b) - v->channels.b;
        float delta_g = (float)(u->channels.g) - v->channels.g;
        float delta_r = (float)(u->channels.r) - v->channels.r;
        return (uint)               (0.299f * delta_b * delta_b +
                                     0.587f * delta_g * delta_g +
                                     0.114f * delta_r * delta_r);
    #else
        int delta_b = (int)(u->channels.b) - v->channels.b;
        int delta_g = (int)(u->channels.g) - v->channels.g;


        int delta_r = (int)(u->channels.r) - v->channels.r;
        return delta_b * delta_b + delta_g * delta_g + delta_r * delta_r;
    #endif
}

inline void WriteColors444(__global uchar* block, const Color * color0, const Color * color1) {


    
    block[0] = (color0->channels.r & 0xf0) | (color1->channels.r >> 4);
    block[1] = (color0->channels.g & 0xf0) | (color1->channels.g >> 4);


    block[2] = (color0->channels.b & 0xf0) | (color1->channels.b >> 4);
}

inline void WriteColors555(__global uchar* block, const Color * color0, const Color * color1) {
    
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
            (short)(color1->channels.r >> 3) - (color0->channels.r >> 3);
    short delta_g =
            (short)(color1->channels.g >> 3) - (color0->channels.g >> 3);
    short delta_b =
            (short)(color1->channels.b >> 3) - (color0->channels.b >> 3);

    
    block[0] = (color0->channels.r & 0xf8) | two_compl_trans_table[delta_r + 4];


    block[1] = (color0->channels.g & 0xf8) | two_compl_trans_table[delta_g + 4];
    block[2] = (color0->channels.b & 0xf8) | two_compl_trans_table[delta_b + 4];
}

inline void WriteCodewordTable(__global uchar* block, uchar sub_block_id, uchar table) {

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

inline void WriteFlip(__global uchar* block, int flip) {
    block[3] &= ~0x01;
    block[3] |= (uchar)(flip);
}

inline void WriteDiff(__global uchar* block, int diff) {
    block[3] &= ~0x02;
    block[3] |= (uchar)(diff) << 1;
}






inline Color makeColor444(const float* bgr) {
    uchar b4 = round_to_4_bits(bgr[0]);
    uchar g4 = round_to_4_bits(bgr[1]);
    uchar r4 = round_to_4_bits(bgr[2]);
    Color bgr444;
    bgr444.channels.b = (b4 << 4) | b4;
    bgr444.channels.g = (g4 << 4) | g4;
    bgr444.channels.r = (r4 << 4) | r4;
    
    bgr444.channels.a = 0x44;
    return bgr444;
}





inline Color makeColor555(const float* bgr) {
    uchar b5 = round_to_5_bits(bgr[0]);
    uchar g5 = round_to_5_bits(bgr[1]);
    uchar r5 = round_to_5_bits(bgr[2]);
    Color bgr555;
    bgr555.channels.b = (b5 > 2);
    bgr555.channels.g = (g5 > 2);
    bgr555.channels.r = (r5 > 2);
    
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

unsigned long computeLuminance(__global uchar* block, const Color* src, const Color * base, int sub_block_id, __constant uchar* idx_to_num_tab, unsigned long threshold){
    uint best_tbl_err = threshold;
    uchar best_tbl_idx = 0;
    uchar best_mod_idx[8][8];  

    
    
    for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
    
    
        Color candidate_color[4];  
        for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
            short lum = g_codeword_tables[tbl_idx][mod_idx];
            candidate_color[mod_idx] = makeColor(base, lum);
        }

        uint tbl_err = 0;

        for (unsigned int i = 0; i < 8; ++i) {
            
            
            uint best_mod_err = threshold;
            for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
                const Color color = candidate_color[mod_idx];

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


int tryCompressSolidBlock(__global uchar *dst, const Color *src, unsigned long *error) {
    for (unsigned int i = 1; i < 16; ++i) {
        if (src[i].bits != src[0].bits)
            return 0;
    }

    
    memset(dst, 0, 8);

    float src_color_float[3] = {(float) (src->channels.b),
                                (float) (src->channels.g),
                                (float) (src->channels.r)};
    Color base = makeColor555(src_color_float);

    WriteDiff(dst, 1);
    WriteFlip(dst, 0);
    WriteColors555(dst, &base, &base);

    uchar best_tbl_idx = 0;
    uchar best_mod_idx = 0;
    uint best_mod_err = UINT32_MAX;

    
    
    for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
        
        
        for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
            short lum = g_codeword_tables[tbl_idx][mod_idx];
            const Color color = makeColor(&base, lum);

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
    for (unsigned int i = 0; i < 2; ++i) {
        for (unsigned int j = 0; j < 8; ++j) {
            
            int texel_num = g_idx_to_num[i][j];
            pix_data |= msb << (texel_num + 16);
            pix_data |= lsb << (texel_num);
        }
    }

    WritePixelData(dst, pix_data);
    *error = 16 * best_mod_err;
    return 1;
}

unsigned long compressBlock(__global uchar *dst, const Color *ver_src, const Color *hor_src, unsigned long threshold) {
    unsigned long solid_error = 0;
    if (tryCompressSolidBlock(dst, ver_src, &solid_error)) {
        return solid_error;
    }

    const Color *sub_block_src[4] = {ver_src, ver_src + 8, hor_src, hor_src + 8};

    Color sub_block_avg[4];
    int use_differential[2] = {1, 1};

    
    
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


    for (unsigned int i = 0; i < 4; ++i) {
        for (unsigned int j = 0; j < 8; ++j) {
            sub_block_err[i] += getColorError(&sub_block_avg[i], &sub_block_src[i][j]);
        }
    }

    int flip = sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1];

    
    memset(dst, 0, 8);

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

    unsigned long lumi_error1 = 0, lumi_error2 = 0;

    
    lumi_error1 = computeLuminance(dst, sub_block_src[sub_block_off_0],
                                   &sub_block_avg[sub_block_off_0], 0,
                                   g_idx_to_num[sub_block_off_0],
                                   threshold);
    
    lumi_error2 = computeLuminance(dst, sub_block_src[sub_block_off_1],
                                   &sub_block_avg[sub_block_off_1], 1,
                                   g_idx_to_num[sub_block_off_1],
                                   threshold);

    return 0;
}

__kernel void compress(__global const uchar *src, __global uchar *dst, int width, int height) {
    int gid_0 = get_global_id(1);
    int gid_1 = get_global_id(0);
    union Color ver_blocks[16];
    union Color hor_blocks[16];

    unsigned long compressed_error = 0;
    int y = gid_0 * 4;
    int x = gid_1 * 4;
    src += gid_0 * 16 * width;
    dst += width * 2 * gid_0 + 8 * gid_1;

    __global union Color *row0 = (__global union Color *)(src + x * 4);
    __global union Color *row1 = row0 + width;
    __global union Color *row2 = row1 + width;
    __global union Color *row3 = row2 + width;

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
