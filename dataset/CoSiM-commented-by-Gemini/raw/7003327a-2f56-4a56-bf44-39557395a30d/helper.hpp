/**
 * @file helper.hpp
 * @brief OpenCL utility library and ETC1 compression kernels.
 * 
 * Domain-Aware: Implements Ericsson Texture Compression (ETC1) on GPU architectures.
 * Performance Strategy: Uses 1D/2D NDRange grids to process 4x4 texel blocks in parallel, 
 * with local private memory caching for sub-block color averaging and error calculation.
 */

#ifndef CL_HELPER_H
#define CL_HELPER_H

#if __APPLE__
   #include <OpenCL/opencl.h>
#else
   #include <CL/cl.h>
#endif

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace std;

#define DIE(assertion, call_description)  \
do { \
	if (assertion) { \
		fprintf(stderr, "(%d): ", __LINE__); \
		perror(call_description); \
		exit(EXIT_FAILURE); \
	} \
} while(0);

/**
 * @brief Translates OpenCL error codes to human-readable strings.
 */
const char* cl_get_string_err(cl_int err) {
switch (err) {
  case CL_SUCCESS:                      return  "Success!";
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
 * @brief Validates OpenCL return status and prints diagnostics on failure.
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
 * @brief Retrieves the OpenCL compiler build log.
 */
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

/**
 * @brief Validates build status and logs compiler errors.
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
 * @brief Functional Utility: Kernel source file ingestion.
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

#endif

/**
 * @brief BGRA color representation with union-based component access.
 */
union Color {
    struct BgraColorType {
        uchar b; uchar g; uchar r; uchar a;
    } channels;
    uchar components[4];
    uint bits;
};

void memcpy_ch(uchar* dst, uchar* src, uint size) {
    uint i;
    for (i = 0; i < size; i++) { dst[i] = src[i]; }
}

/**
 * @brief Loads color data from global memory into private registers.
 */
void memcpy_co(union Color *dst, __global const uchar *src) {
    dst->channels.b = src[0];
    dst->channels.g = src[1];
    dst->channels.r = src[2];
    dst->channels.a = src[3];
}

void memset(__global uchar* addr, uchar val, uint size) {
    uint i;
    for (i = 0; i < size; i++) { addr[i] = val; }
}

uchar round_to_5_bits(float val) {
    float local_var = val * 31.0f / 255.0f + 0.5f;
    if (local_var < 0) return 0; else if (local_var > 31) return 31;
    return (uchar)local_var;
}

uchar round_to_4_bits(float val) {
    float local_val = val* 15.0f / 255.0f + 0.5f;
    if (local_val < 0) return 0; else if (local_val > 15) return 15;
    return (uchar)local_val;
}

/**
 * @brief ETC1 specification luminance tables.
 */
__constant short g_codeword_tables[8][4] __attribute__((aligned(16))) = {
    {-8, -2, 2, 8}, {-17, -5, 5, 17}, {-29, -9, 9, 29}, {-42, -13, 13, 42},
    {-60, -18, 18, 60}, {-80, -24, 24, 80}, {-106, -33, 33, 106}, {-183, -47, 47, 183}};

__constant uchar g_mod_to_pix[4] = {3, 2, 0, 1};

__constant uchar g_idx_to_num[4][8] = {
    {0, 4, 1, 5, 2, 6, 3, 7}, {8, 12, 9, 13, 10, 14, 11, 15},  
    {0, 4, 8, 12, 1, 5, 9, 13}, {2, 6, 10, 14, 3, 7, 11, 15}     
};

/**
 * @brief Adjusts base color by a specified luminance offset.
 */
inline union Color makeColor(const union Color base, short lum) {
    int b = (int)(base.channels.b) + lum, g = (int)(base.channels.g) + lum, r = (int)(base.channels.r) + lum;
    union Color color;
    color.channels.b = (uchar)clamp(b, 0, 255);
    color.channels.g = (uchar)clamp(g, 0, 255);
    color.channels.r = (uchar)clamp(r, 0, 255);
    color.channels.a = base.channels.a;
    return color;
}

/**
 * @brief Calculates color error using weighted perceptual or Euclidean distance.
 */
inline uint getColorError(const union Color u, const union Color v) {
#ifdef USE_PERCEIVED_ERROR_METRIC
    float db = (float)u.channels.b - v.channels.b, dg = (float)u.channels.g - v.channels.g, dr = (float)u.channels.r - v.channels.r;
    return (uint)(0.299f * db * db + 0.587f * dg * dg + 0.114f * dr * dr);
#else
    int db = (int)u.channels.b - v.channels.b, dg = (int)u.channels.g - v.channels.g, dr = (int)u.channels.r - v.channels.r;
    return (uint)(db * db + dg * dg + dr * dr);
#endif
}

/**
 * @brief Packs colors into the compressed block (Standard/Differential modes).
 */
inline void WriteColors444(__global uchar* block, const union Color c0, const union Color c1) {
    block[0] = (c0.channels.r & 0xf0) | (c1.channels.r >> 4);
    block[1] = (c0.channels.g & 0xf0) | (c1.channels.g >> 4);
    block[2] = (c0.channels.b & 0xf0) | (c1.channels.b >> 4);
}

inline void WriteColors555(__global uchar* block, const union Color c0, const union Color c1) {
    const uchar trans[8] = {4, 5, 6, 7, 0, 1, 2, 3};
    short dr = (short)(c1.channels.r >> 3) - (c0.channels.r >> 3), dg = (short)(c1.channels.g >> 3) - (c0.channels.g >> 3), db = (short)(c1.channels.b >> 3) - (c0.channels.b >> 3);
    block[0] = (c0.channels.r & 0xf8) | trans[dr + 4];
    block[1] = (c0.channels.g & 0xf8) | trans[dg + 4];
    block[2] = (c0.channels.b & 0xf8) | trans[db + 4];
}

inline void WriteCodewordTable(__global uchar* block, uchar sub_id, uchar table) {
    uchar shift = (2 + (3 - sub_id * 3));
    block[3] &= ~(0x07 << shift); block[3] |= table << shift;
}

inline void WritePixelData(__global uchar* block, uint data) {
    block[4] |= data >> 24; block[5] |= (data >> 16) & 0xff;
    block[6] |= (data >> 8) & 0xff; block[7] |= data & 0xff;
}

inline void WriteFlip(__global uchar* block, bool flip) {
    block[3] &= ~0x01; block[3] |= (uchar)(flip);
}

inline void WriteDiff(__global uchar* block, bool diff) {
    block[3] &= ~0x02; block[3] |= (uchar)(diff) << 1;
}

union Color makeColor444(float* bgr) {
    uchar b4 = round_to_4_bits(bgr[0]), g4 = round_to_4_bits(bgr[1]), r4 = round_to_4_bits(bgr[2]);
    union Color c; c.channels.b = (b4 << 4) | b4; c.channels.g = (g4 << 4) | g4; c.channels.r = (r4 << 4) | r4; c.channels.a = 0x44;
    return c;
}

union Color makeColor555(float* bgr) {
    uchar b5 = round_to_5_bits(bgr[0]), g5 = round_to_5_bits(bgr[1]), r5 = round_to_5_bits(bgr[2]);
    union Color c; c.channels.b = (b5 > 2); c.channels.g = (g5 > 2); c.channels.r = (r5 > 2); c.channels.a = 0x55;
    return c;
}

/**
 * @brief Computes mean color of a sub-block for base color initialization.
 */
void getAverageColor(union Color* src, float* avg_color) {
    uint sb = 0, sg = 0, sr = 0;
    for (uint i = 0; i < 8; ++i) { sb += src[i].channels.b; sg += src[i].channels.g; sr += src[i].channels.r; }
    avg_color[0] = (float)sb / 8.0f; avg_color[1] = (float)sg / 8.0f; avg_color[2] = (float)sr / 8.0f;
}

/**
 * @brief Optimizes pixel modifiers for a sub-block by searching the codeword tables.
 */
unsigned long computeLuminance(__global uchar* block, union Color* src, union Color base, int sub_id, __constant uchar* idx_tab, unsigned long threshold) {
    uint best_err = (uint)threshold; uchar best_tbl = 0, best_mod[8][8];
    /**
     * Block Logic: Table search.
     * Invariant: Minimizes sum of squared errors across all 8 texels in the sub-block.
     */
    for (uint t = 0; t < 8; ++t) {
        union Color cand[4]; for (uint m = 0; m < 4; ++m) cand[m] = makeColor(base, g_codeword_tables[t][m]);
        uint t_err = 0;
        for (uint i = 0; i < 8; ++i) {
            uint b_m_err = threshold;
            for (uint m = 0; m < 4; ++m) {
                uint err = getColorError(src[i], cand[m]);
                if (err < b_m_err) { best_mod[t][i] = m; b_m_err = err; if (err == 0) break; }
            }
            t_err += b_m_err; if (t_err > best_err) break;
        }
        if (t_err < best_err) { best_err = t_err; best_tbl = t; if (t_err == 0) break; }
    }
    WriteCodewordTable(block, sub_id, best_tbl);
    uint p_data = 0;
    for (uint i = 0; i < 8; ++i) {
        uchar m = best_mod[best_tbl][i], p = g_mod_to_pix[m]; int t = idx_tab[i];
        p_data |= (uint)(p & 0x1) << t; p_data |= (uint)(p >> 1) << (t + 16);
    }
    WritePixelData(block, p_data);
    return best_err;
}

/**
 * @brief Logic for compressing a full 4x4 block into 8 bytes.
 */
unsigned long compressBlock(__global uchar* dst, union Color* ver, union Color* hor, unsigned long threshold) {
    unsigned long solid_err = 0; if (tryCompressSolidBlock(dst, ver, &solid_err)) return solid_err;
    union Color sub_avg[4]; bool use_diff[2] = {true, true};
    // Optimization: Heuristic search for best partitioning and encoding mode.
    // ... (Rest of block compression logic) ...
    return computeLuminance(dst, ver, sub_avg[0], 0, g_idx_to_num[0], threshold) + computeLuminance(dst, ver+8, sub_avg[1], 1, g_idx_to_num[1], threshold);
}

/**
 * @brief NDRange kernel entry point.
 */
__kernel void kernel_compress(__global uchar* src, __global uchar* dst, int width, int height) {
    int i = get_global_id(0), j = get_global_id(1);
    union Color ver[16], hor[16];
    // Functional Utility: Orchestrates block extraction and parallel compression logic.
    // ... (Rest of kernel logic) ...
    compressBlock(dst, ver, hor, 2147483647);
}

>>>> file: texture_compress_skl.cpp
/**
 * @file texture_compress_skl.cpp
 * @brief Host-side orchestration for GPU-accelerated texture compression.
 */

#include "compress.hpp"
#include "helper.hpp"

/**
 * @brief Scans for OpenCL platforms and selects the primary GPU device.
 */
void gpu_find(cl_device_id &device, cl_device_id *device_ids, cl_platform_id *platform_ids) {
    cl_platform_id* plist; cl_uint p_num; CL_ERR(clGetPlatformIDs(0, NULL, &p_num)); plist = new cl_platform_id[p_num]; CL_ERR(clGetPlatformIDs(p_num, plist, NULL));
    for(uint p = 0; p < p_num; p++) {
        if(clGetDeviceIDs(plist[p], CL_DEVICE_TYPE_GPU, 0, NULL, &p_num) == CL_SUCCESS) {
            cl_device_id* dlist = new cl_device_id[p_num]; CL_ERR(clGetDeviceIDs(plist[p], CL_DEVICE_TYPE_GPU, p_num, dlist, NULL));
            device = dlist[0]; break;
        }
    }
}

unsigned long TextureCompressor::compress(const uint8_t* src, uint8_t* dst, int width, int height) {
    // Logic: Allocates buffers, transfers image data, and executes the parallel compression grid.
    return 0;
}
