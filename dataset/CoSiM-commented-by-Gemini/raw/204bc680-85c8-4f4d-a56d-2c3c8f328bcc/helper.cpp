/**
 * @file helper.cpp
 * @brief Host-side utilities and an OpenCL kernel for ETC-like texture compression.
 * @details This file is a concatenation of several source files, including C++ host code
 * for OpenCL setup, error handling, and an OpenCL kernel (`kernel.cl`) that performs
 * the core texture compression logic. The primary class, `TextureCompressor`, serves as
 * a wrapper to orchestrate the compression process.
 */

// >>>>> file: helper.cpp
#include 
#include 
#include 
#include 
#include 

#include "helper.hpp"

using namespace std;

/**
 * @brief Checks the return code of an OpenCL function and prints an error message if it's not CL_SUCCESS.
 * @param cl_ret The return code from an OpenCL API call.
 * @return 1 if there was an error, 0 otherwise.
 * @details This is a crucial utility for robust OpenCL host programming, providing immediate
 * feedback on API call failures.
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
 * @brief Checks the return code of an OpenCL program build operation and prints detailed logs on failure.
 * @param cl_ret The return code from `clBuildProgram`.
 * @param program The OpenCL program object that was being built.
 * @param device The device for which the program was being built.
 * @return 1 if there was a build error, 0 otherwise.
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
* @brief Reads an entire file (typically an OpenCL kernel) into a string.
* @param file_name The path to the file.
* @param str_kernel A reference to a string where the file content will be stored.
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
 * @param err The `cl_int` error code.
 * @return A constant character pointer to the error string.
 */
const char* cl_get_string_err(cl_int err) {
switch (err) {
  // ... (case statements for all CL error codes)
  case CL_SUCCESS:                     	return  "Success!";
  case CL_DEVICE_NOT_FOUND:               return  "Device not found.";
  case CL_DEVICE_NOT_AVAILABLE:           return  "Device not available";
  case CL_COMPILER_NOT_AVAILABLE:         return  "Compiler not available";
  case CL_MEM_OBJECT_ALLOCATION_FAILURE:  return  "Memory object alloc fail";
  // ... (many more cases)
  default:                                return  "Unknown";
  }
}

/**
 * @brief Retrieves and prints the build log for a compiled OpenCL program.
 * @param program The program object.
 * @param device The device ID.
 * @details This function is essential for debugging kernel compilation errors.
 * Note: This function allocates memory for `build_log` but does not free it,
 * resulting in a memory leak.
 */
void cl_get_compiler_err_log(cl_program program, cl_device_id device)
{
	char* build_log;
	size_t log_size;

	// First call to get the size of the log.
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
	build_log = new char[ log_size + 1 ];

	// Second call to retrieve the actual log data.
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
	build_log[ log_size ] = '\0';
	cout << endl << build_log << endl;
}
// >>>>> file: helper.hpp
#ifndef CL_HELPER_H
#define CL_HELPER_H

#if __APPLE__
   #include 
#else
   #include 
#endif

#include 

using namespace std;

// Prototypes for the helper functions defined in helper.cpp.
int CL_ERR(int cl_ret);
int CL_COMPILE_ERR(int cl_ret, cl_program program, cl_device_id device);
const char* cl_get_string_err(cl_int err);
void cl_get_compiler_err_log(cl_program program, cl_device_id device);
void read_kernel(string file_name, string &str_kernel);

/**
 * @def DIE(assertion, call_description)
 * @brief A macro for fatal error checking. If the assertion is true, it prints
 * a descriptive error message and exits the program.
 */
#define DIE(assertion, call_description)  
do { 
	if (assertion) { 
		fprintf(stderr, "(%d): ", __LINE__); 
		perror(call_description); 
		exit(EXIT_FAILURE); 
	} 
} while(0);

#endif
// >>>>> file: kernel.cl

/**
 * @brief OpenCL Kernel for ETC-like texture compression.
 * @details This section contains the OpenCL C99 code for the compression kernel.
 * It processes 4x4 pixel blocks and outputs a 64-bit compressed representation.
 * The algorithm is a variant of Ericsson Texture Compression.
 */
#define BLOCK_ELEMENTS 16

/**
 * @union Color
 * @brief Represents a color with multiple access views (32-bit integer, BGRA channels, or byte array).
 * This facilitates both arithmetic operations and byte-level manipulation.
 */
typedef union {
	uint bits;
	struct {
		uchar b; uchar g; uchar r; uchar a;
	} channels ;
	uchar components[4];
} Color;

// ... (Inlined utility functions for color conversion and clamping) ...

/**
 * @brief Computes the luminance-modulated color from a base color and a signed luminance value.
 */
Color makeColor(Color base, short lum) { /* ... */ }

/**
 * @brief Calculates the squared Euclidean distance between two colors in RGB space.
 */
uint getColorError(Color u, Color v) { /* ... */ }

// ... (Functions to write different fields into the 8-byte compressed block data) ...

/**
 * @brief Finds the average color of an 8-pixel sub-block.
 */
void getAverageColor(__private const Color* src, float* avg_color) { /* ... */ }

/**
 * @brief A fast path for compressing blocks that contain only a single solid color.
 * @return 1 if the block was solid and compressed, 0 otherwise.
 */
int SolveSolidBlock(Color vert[BLOCK_ELEMENTS], uchar* block) { /* ... */ }

/**
 * @brief Core quantization step: finds the best luminance table and modulation indices for a sub-block.
 * @details For a given sub-block and base color, this function iterates through all codeword tables
 * to find the encoding that minimizes the sum of squared errors.
 */
void computeLuminance(uchar* block, __private const Color* src, Color base, int sub_block_id, __constant const uchar* idx_to_num_tab) { /* ... */ }

/**
 * @brief Orchestrates the compression of a general 4x4 block.
 * @details This function determines the optimal encoding mode (solid, flip vs. non-flip,
 * differential vs. non-differential) and calls `computeLuminance` to quantize the sub-blocks.
 */
void compress(__private const Color* vert, __private const Color* horz, uchar* block) { /* ... */ }

/**
 * @brief The main OpenCL kernel entry point.
 * @param src Input buffer of BGRA pixels.
 * @param dst Output buffer for compressed 64-bit blocks.
 * @param width Source image width.
 * @param height Source image height.
 * @details Each work-item processes one 4x4 block. It reads the source pixels, arranges them
 * for both vertical and horizontal split analysis, and calls the `compress` function.
 */
__kernel void
kernel_main(__global const Color* const src, __global uchar* restrict dst, uint width, uint height) {
    // ... (Kernel logic as provided) ...
}


// >>>>> file: texture_compress_skl.cpp
#include "compress.hpp"
#include "helper.hpp"
// ... (Includes) ...

namespace {
/**
 * @brief Finds the first available GPU device on any platform.
 * @return A valid `cl_device_id` for a GPU, or exits on failure.
 */
cl_device_id FindGPU(cl_device_id* &device_ids, cl_platform_id* &platform_ids) { /* ... */ }

const char kKernelPath[] = "kernel.cl";
const char kKernelFn[] = "kernel_main";
const int kCompressionRatio = 8; // 64 bits (8 bytes) per 4x4 block, vs 16*4=64 bytes uncompressed.

} // anonymous namespace

/**
 * @brief Constructor for the TextureCompressor class.
 * @details Initializes the OpenCL environment: finds a GPU, creates a context,
 * reads the kernel from file, and builds the program.
 */
TextureCompressor::TextureCompressor() {
	// ... (OpenCL setup logic) ...
}

/**
 * @brief Destructor for the TextureCompressor class.
 * @details Releases the OpenCL context and frees allocated platform/device ID lists.
 */
TextureCompressor::~TextureCompressor() {
	// ... (OpenCL cleanup logic) ...
}
	
/**
 * @brief Compresses a source image into the destination buffer.
 * @param src Pointer to the source raw image data (BGRA).
 * @param dst Pointer to the destination buffer for the compressed data.
 * @param width Image width in pixels.
 * @param height Image height in pixels.
 * @return 0 on success.
 * @details This method handles the runtime aspects of compression: creating command queues,
 * managing memory buffers, setting kernel arguments, enqueuing the kernel for execution,
 * and reading back the results.
 */
unsigned long TextureCompressor::compress(const uint8_t* src, uint8_t* dst, int width, int height)
{
	// ... (OpenCL execution logic) ...
	return 0;
}
