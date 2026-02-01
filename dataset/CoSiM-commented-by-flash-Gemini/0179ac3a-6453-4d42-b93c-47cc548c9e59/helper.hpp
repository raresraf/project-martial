/**
 * @file helper.hpp
 * @brief This header file provides a collection of utility functions and macros
 *        for OpenCL-based GPU programming, primarily focused on texture compression.
 *        It includes error handling, kernel loading, color manipulation, and
 *        block compression algorithms.
 *
 * This file is designed to support high-performance computing tasks on GPUs,
 * especially for image processing and compression, by providing essential
 * OpenCL API wrappers and data structures.
 */

#include 
#include 
#include 
#include 
#include 

#ifndef CL_HELPER_H
#define CL_HELPER_H

#if __APPLE__
   #include 
#else
   #include 
#endif

/**
 * @brief Macro for robust error checking and program termination.
 *
 * This macro checks an assertion; if it's true, it prints the current
 * line number, a system error message (if available), and terminates
 * the program with an error status.
 *
 * @param assertion The boolean expression to check. If true, the program exits.
 * @param call_description A string describing the context of the call, used in the error message.
 */
#define DIE(assertion, call_description)  \
do { \
	if (assertion) { \
		fprintf(stderr, "(%d): ", __LINE__); \
		perror(call_description); \
		exit(EXIT_FAILURE); \
	} \
} while(0);

using namespace std;

/**
 * @brief Checks the return value of an OpenCL API call and prints an error message if it's not CL_SUCCESS.
 * @param cl_ret The integer return code from an OpenCL API function.
 * @return 1 if an error occurred (cl_ret != CL_SUCCESS), 0 otherwise.
 */
int CL_ERR(int cl_ret);

/**
 * @brief Checks the return value of an OpenCL program compilation.
 *        If compilation failed, it prints the error and retrieves the compiler build log.
 * @param cl_ret The integer return code from clBuildProgram.
 * @param program The OpenCL program object.
 * @param device The OpenCL device ID.
 * @return 1 if a compilation error occurred, 0 otherwise.
 */
int CL_COMPILE_ERR(int cl_ret, cl_program program, cl_device_id device);

/**
 * @brief Reads the content of an OpenCL kernel file into a string.
 * @param file_name The name of the kernel file to read.
 * @param str_kernel A reference to a string where the kernel source will be stored.
 * Functional Utility: Facilitates loading OpenCL kernel source code from external files.
 */
void read_kernel(string file_name, string &str_kernel);

/**
 * @brief Retrieves a human-readable string for an OpenCL error code.
 * @param err The OpenCL error code (cl_int).
 * @return A constant character string describing the error.
 */
const char* cl_get_string_err(cl_int err);

/**
 * @brief Retrieves and prints the OpenCL compiler build log for a given program and device.
 * Functional Utility: Essential for debugging OpenCL kernel compilation failures.
 * @param program The OpenCL program object.
 * @param device The OpenCL device ID.
 */
void cl_get_compiler_err_log(cl_program program, cl_device_id device);


/**
 * @brief Checks the return value of an OpenCL API call and prints an error message if it's not CL_SUCCESS.
 * @param cl_ret The integer return code from an OpenCL API function.
 * @return 1 if an error occurred (cl_ret != CL_SUCCESS), 0 otherwise.
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
 * @brief Checks the return value of an OpenCL program compilation.
 *        If compilation failed, it prints the error and retrieves the compiler build log.
 * @param cl_ret The integer return code from clBuildProgram.
 * @param program The OpenCL program object.
 * @param device The OpenCL device ID.
 * @return 1 if a compilation error occurred, 0 otherwise.
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
 * @brief Reads the content of an OpenCL kernel file into a string.
 * @param file_name The name of the kernel file to read.
 * @param str_kernel A reference to a string where the kernel source will be stored.
 * Functional Utility: Enables loading OpenCL kernel source code from external files,
 *                     crucial for flexible kernel management.
 * Precondition: `file_name` must point to a valid and readable file.
 * Postcondition: `str_kernel` will contain the full content of the file.
 * Throws: Exits the program if the file cannot be opened.
 */
void read_kernel(string file_name, string &str_kernel)
{
	ifstream in_file(file_name.c_str());
	// Block Logic: Attempts to open the specified kernel file.
	// Invariant: `in_file` must be successfully opened to proceed.
	in_file.open(file_name.c_str());
	DIE( !in_file.is_open(), "ERR OpenCL kernel file. Same directory as binary ?" );

	stringstream str_stream;
	str_stream << in_file.rdbuf(); // Read entire file buffer into stringstream.

	str_kernel = str_stream.str(); // Extract string from stringstream.
}


/**
 * @brief Retrieves a human-readable string for an OpenCL error code.
 * Functional Utility: Maps numerical OpenCL error codes to descriptive messages,
 * facilitating debugging and user feedback.
 * @param err The OpenCL error code (cl_int).
 * @return A constant character string describing the error.
 */
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
 * @brief Retrieves and prints the OpenCL compiler build log for a given program and device.
 * Functional Utility: Provides essential diagnostic information for debugging kernel compilation issues,
 *                     outputting the log to the standard output.
 * @param program The OpenCL program object for which to retrieve the build log.
 * @param device The OpenCL device ID on which the program was built.
 */
void cl_get_compiler_err_log(cl_program program, cl_device_id device)
{
	char* build_log;
	size_t log_size;

	// Query the size of the build log first.
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
						  0, NULL, &log_size);
	// Allocate memory for the build log.
	build_log = new char[ log_size + 1 ];

	// Retrieve the actual build log message.
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
						  log_size, build_log, NULL);
	build_log[ log_size ] = '\0'; // Null-terminate the string.
	cout << endl << build_log << endl; // Print the build log.
}

#endif // CL_HELPER_H

/**
 * @brief Alignment specifier for data structures to ensure optimal memory access patterns,
 *        especially for SIMD operations or cache line alignment.
 * @param X The alignment boundary in bytes (e.g., 16 for 128-bit alignment).
 */
#define ALIGNAS(X)	__attribute__((aligned(X)))

/** @brief Maximum value for an unsigned 32-bit integer. */
#define UINT32_MAX  (0xffffffff)
/** @brief Maximum value for a signed 32-bit integer. */
#define INT32_MAX   (0x7fffffff)
/** @brief Defines the width of a processing block, often used in texture compression. */
#define BLOCKWIDTH   4
/** @brief Defines the total size of a processing block in terms of elements, e.g., 4x4 pixels = 16. */
#define BLOCKSIZE   16

/**
 * @union Color
 * @brief Represents a color with byte-level access to individual channels (BGRA) or as a 32-bit integer.
 * Functional Utility: Provides flexible manipulation of pixel data, allowing access to individual
 *                     color components or the entire pixel as a single integer for efficiency.
 */
typedef union Color {
	/** @struct BgraColorType
	 * @brief Structure for accessing color channels individually.
	 */
	struct BgraColorType {
		uchar b; /**< Blue channel component. */
		uchar g; /**< Green channel component. */
		uchar r; /**< Red channel component. */
		uchar a; /**< Alpha channel component. */
	} channels;
    
	uchar components[4]; /**< Array access to color components (B, G, R, A). */
	uint bits; /**< 32-bit integer representation of the color. */
} Color;


/**
 * @brief Global constant table for various codeword sets used in texture compression.
 * Memory Usage: `__constant` indicates this data resides in constant memory, accessible
 *               to all work-items but read-only after initialization.
 * Functional Utility: Stores predefined luminance or color difference values for different
 *                     compression tables, indexed by table and codeword ID.
 */
ALIGNAS(16) __constant short g_codeword_tables[8][4] = {
	{-8, -2, 2, 8},
	{-17, -5, 5, 17},
	{-29, -9, 9, 29},
	{-42, -13, 13, 42},
	{-60, -18, 18, 60},
	{-80, -24, 24, 80},
	{-106, -33, 33, 106},
	{-183, -47, 47, 183}};
    
/**
 * @brief Global constant mapping from a modification index to a pixel index.
 * Memory Usage: `__constant` indicates this data resides in constant memory.
 * Functional Utility: Used in pixel data encoding for texture compression.
 */
__constant uchar g_mod_to_pix[4] = {3, 2, 0, 1};

/**
 * @brief Global constant table for mapping an index to a number, possibly for reordering pixel data.
 * Memory Usage: `__constant` indicates this data resides in constant memory.
 * Functional Utility: Provides reordering indices for pixel data within a block,
 *                     relevant for different sub-block configurations or shuffling.
 */
__constant uchar g_idx_to_num[4][8] = {
	{0, 4, 1, 5, 2, 6, 3, 7},
	{8, 12, 9, 13, 10, 14, 11, 15},
	{0, 4, 8, 12, 1, 5, 9, 13},
	{2, 6, 10, 14, 3, 7, 11, 15}
};

/**
 * @brief Clamps an unsigned character value within a specified minimum and maximum range.
 * @param val The value to clamp.
 * @param min The minimum allowed value.
 * @param max The maximum allowed value.
 * @return The clamped unsigned character value.
 */
inline uchar clamp_uchar(uchar val, uchar min, uchar max) {
	// Block Logic: If `val` is less than `min`, returns `min`. If `val` is greater than `max`, returns `max`. Otherwise, returns `val`.
	return val < min ? min : (val > max ? max : val);
}

/**
 * @brief Clamps an integer value within a specified minimum and maximum range.
 * @param val The value to clamp.
 * @param min The minimum allowed value.
 * @param max The maximum allowed value.
 * @return The clamped integer value.
 */
inline int clamp_int(int val, int min, int max) {
	// Block Logic: If `val` is less than `min`, returns `min`. If `val` is greater than `max`, returns `max`. Otherwise, returns `val`.
	return val < min ? min : (val > max ? max : val);
}

/**
 * @brief Rounds a float value (0-255 range) to a 5-bit unsigned character representation.
 * Functional Utility: Converts an 8-bit color component to a 5-bit representation,
 *                     typical in some texture formats (e.g., RGB565 or BC1/DXT1).
 * @param val The float value to round.
 * @return The rounded 5-bit unsigned character value.
 */
inline uchar round_to_5_bits(float val) {
	return clamp_uchar(val * 31.0f / 255.0f + 0.5f, 0, 31);
}

/**
 * @brief Rounds a float value (0-255 range) to a 4-bit unsigned character representation.
 * Functional Utility: Converts an 8-bit color component to a 4-bit representation,
 *                     typical in some texture formats (e.g., BC1/DXT1 alpha or some color modes).
 * @param val The float value to round.
 * @return The rounded 4-bit unsigned character value.
 */
inline uchar round_to_4_bits(float val) {
	return clamp_uchar(val * 15.0f / 255.0f + 0.5f, 0, 15);
}

/**
 * @brief Creates a new Color by adding a luminance offset to a base color.
 * @param base The base color.
 * @param lum The luminance offset to apply.
 * @return A new Color object with the luminance adjusted channels, clamped to [0, 255].
 */
inline Color makeColor(const Color base, short lum) {
    Color color;
    
	// Apply luminance to each color channel.
	int b = (int)(base.channels.b) + lum;
	int g = (int)(base.channels.g) + lum;
	int r = (int)(base.channels.r) + lum;
    
	// Clamp channel values to the valid 0-255 range.
	color.channels.b = (uchar)(clamp_int(b, 0, 255));
	color.channels.g = (uchar)(clamp_int(g, 0, 255));
	color.channels.r = (uchar)(clamp_int(r, 0, 255));
    
	return color;
}

/**
 * @brief Calculates the squared color error between two colors.
 * Functional Utility: Quantifies the perceptual or simple squared Euclidean distance
 *                     between two colors, used for optimization in compression algorithms.
 * @param u The first color.
 * @param v The second color.
 * @return The squared color error. Uses a perceived error metric if `USE_PERCEIVED_ERROR_METRIC` is defined.
 */
inline int getColorError(const Color u, const Color v) {
#ifdef USE_PERCEIVED_ERROR_METRIC
	// Performance Optimization: Uses a weighted squared error for a more perceptually accurate distance.
	float delta_b = (float)(u.channels.b) - v.channels.b;
	float delta_g = (float)(u.channels.g) - v.channels.g;
	float delta_r = (float)(u.channels.r) - v.channels.r;
	return (int)(0.299f * delta_b * delta_b +
               0.587f * delta_g * delta_g +
			   0.114f * delta_r * delta_r);
#else
	// Standard Euclidean squared distance in RGB space.
	int delta_b = (int)(u.channels.b) - v.channels.b;
	int delta_g = (int)(u.channels.g) - v.channels.g;
	int delta_r = (int)(u.channels.r) - v.channels.r;

	return delta_b * delta_b + delta_g * delta_g + delta_r * delta_r;
#endif
}

/**
 * @brief Writes two 4:4:4 color values into a compressed block.
 * Memory Usage: `__global` indicates `block` is in global memory.
 * Functional Utility: Encodes two 24-bit RGB colors (reduced to 12 bits each)
 *                     into a compact 3-byte representation within the output block.
 * @param block Pointer to the global memory block where compressed data is written.
 * @param color0 The first color (typically 4:4:4 format).
 * @param color1 The second color (typically 4:4:4 format).
 */
inline void WriteColors444(__global uchar *block,
						   const Color color0,
						   const Color color1) {
	// Block Logic: Extracts the most significant 4 bits of each color channel
	//              and packs them into 3 bytes.
	// Red channels: MSB of color0's red and MSB of color1's red.
	block[0] = (color0.channels.r & 0xf0) | (color1.channels.r >> 4);
	// Green channels: MSB of color0's green and MSB of color1's green.
	block[1] = (color0.channels.g & 0xf0) | (color1.channels.g >> 4);
	// Blue channels: MSB of color0's blue and MSB of color1's blue.
	block[2] = (color0.channels.b & 0xf0) | (color1.channels.b >> 4);
}

/**
 * @brief Writes two 5:5:5 color values into a compressed block, often used in differential encoding.
 * Memory Usage: `__global` indicates `block` is in global memory.
 * Functional Utility: Encodes two 15-bit RGB colors (reduced to 5 bits each)
 *                     and their 3-bit differential values into a compact 3-byte representation.
 * @param block Pointer to the global memory block where compressed data is written.
 * @param color0 The base color (typically 5:5:5 format).
 * @param color1 The differential color (typically 5:5:5 format).
 */
inline void WriteColors555(__global uchar* block,
						   const Color color0,
						   const Color color1) {
	
	// Two's complement translation table for 3-bit signed differences.
	const uchar two_compl_trans_table[8] = {
		4,  // -4
		5,  // -3
		6,  // -2
		7,  // -1
		0,  // 0
		1,  // 1
		2,  // 2
		3,  // 3
	};
	
	// Calculate 3-bit differences for each color channel.
	short delta_r =
	(short)(color1.channels.r >> 3) - (color0.channels.r >> 3);
	short delta_g =
	(short)(color1.channels.g >> 3) - (color0.channels.g >> 3);
	short delta_b =
	(short)(color1.channels.b >> 3) - (color0.channels.b >> 3);
	
	// Write the 5-bit base color and 3-bit delta (encoded via lookup table) for each channel.
	block[0] = (color0.channels.r & 0xf8) | two_compl_trans_table[delta_r + 4];
	block[1] = (color0.channels.g & 0xf8) | two_compl_trans_table[delta_g + 4];
	block[2] = (color0.channels.b & 0xf8) | two_compl_trans_table[delta_b + 4];
}

/**
 * @brief Writes a codeword table index into a compressed block.
 * Memory Usage: `__global` indicates `block` is in global memory.
 * Functional Utility: Embeds the selected color/luminance table index into the control
 *                     byte of the compressed block, based on the sub-block ID.
 * @param block Pointer to the global memory block.
 * @param sub_block_id Identifier for the sub-block (0 or 1).
 * @param table The index of the codeword table to write.
 */
inline void WriteCodewordTable(__global uchar* block,
							   uchar sub_block_id,
							   uchar table) {
	// Bitwise Operation: Calculates the shift amount based on the sub-block ID to target specific bits in block[3].
	uchar shift = (2 + (3 - sub_block_id * 3));
	block[3] &= ~(0x07 << shift); // Clear 3 bits for the table index.
	block[3] |= table << shift; // Set the new table index.
}

/**
 * @brief Writes 32-bit pixel data (e.g., color indices) into the compressed block.
 * Memory Usage: `__global` indicates `block` is in global memory.
 * Functional Utility: Packs 32 bits of encoded pixel data into bytes 4-7 of the
 *                     compressed block, typically representing indices into a color palette.
 * @param block Pointer to the global memory block.
 * @param pixel_data The 32-bit pixel data to write.
 */
inline void WritePixelData(__global uchar* block, uint pixel_data) {
	// Bitwise Operation: Extracts and writes individual bytes of the 32-bit pixel data.
	block[4] |= pixel_data >> 24;
	block[5] |= (pixel_data >> 16) & 0xff;
	block[6] |= (pixel_data >> 8) & 0xff;
	block[7] |= pixel_data & 0xff;
}

/**
 * @brief Writes the flip flag into the compressed block.
 * Memory Usage: `__global` indicates `block` is in global memory.
 * Functional Utility: Sets a bit in the control byte to indicate whether the block's
 *                     orientation (e.g., horizontal/vertical split) is flipped.
 * @param block Pointer to the global memory block.
 * @param flip Boolean value for the flip flag.
 */
inline void WriteFlip(__global uchar* block, bool flip) {
	block[3] &= ~0x01; // Clear the LSB.
	block[3] |= (uchar)(flip); // Set the LSB based on `flip`.
}

/**
 * @brief Writes the differential flag into the compressed block.
 * Memory Usage: `__global` indicates `block` is in global memory.
 * Functional Utility: Sets a bit in the control byte to indicate whether the block uses
 *                     differential color encoding (e.g., 5:5:5 vs. 4:4:4).
 * @param block Pointer to the global memory block.
 * @param diff Boolean value for the differential flag.
 */
inline void WriteDiff(__global uchar* block, bool diff) {
	block[3] &= ~0x02; // Clear the second LSB.
	block[3] |= (uchar)(diff) << 1; // Set the second LSB based on `diff`.
}

/**
 * @brief Creates a Color object from BGR float components, rounding to 4-bit precision per channel.
 * Functional Utility: Converts floating-point BGR values (0.0-255.0) into a Color struct
 *                     with 4-bit per channel representation, then expanded to 8-bit for storage.
 * @param bgr Array of 3 floats representing Blue, Green, Red components.
 * @return A Color object with 4-bit rounded BGR channels.
 */
inline Color makeColor444(const float* bgr) {
	uchar b4 = round_to_4_bits(bgr[0]);
	uchar g4 = round_to_4_bits(bgr[1]);
	uchar r4 = round_to_4_bits(bgr[2]);
	Color bgr444;
	// Expand 4-bit value to 8-bit by replicating the 4 bits (e.g., 0xA -> 0xAA).
	bgr444.channels.b = (b4 << 4) | b4;
	bgr444.channels.g = (g4 << 4) | g4;
	bgr444.channels.r = (r4 << 4) | r4;
	
	bgr444.channels.a = 0x44; // Placeholder alpha.
	return bgr444;
}

/**
 * @brief Creates a Color object from BGR float components, rounding to 5-bit precision per channel.
 * Functional Utility: Converts floating-point BGR values (0.0-255.0) into a Color struct
 *                     with 5-bit per channel representation, then stored within an 8-bit `uchar`.
 * @param bgr Array of 3 floats representing Blue, Green, Red components.
 * @return A Color object with 5-bit rounded BGR channels.
 */
inline Color makeColor555(const float* bgr) {
	uchar b5 = round_to_5_bits(bgr[0]);
	uchar g5 = round_to_5_bits(bgr[1]);
	uchar r5 = round_to_5_bits(bgr[2]);
	Color bgr555;
	// Store 5-bit components shifted for compact representation.
	// This usually means the 3 LSBs are 0 or used for something else.
	bgr555.channels.b = (b5 << 3);
	bgr555.channels.g = (g5 << 3);
	bgr555.channels.r = (r5 << 3);
	
	bgr555.channels.a = 0x55; // Placeholder alpha.
	return bgr555;
}

/**
 * @brief Computes the average BGR color components for an array of 8 Color pixels.
 * @param src Pointer to an array of 8 Color structures.
 * @param avg_color Array of 3 floats to store the computed average BGR values.
 * Functional Utility: Calculates the average color of a small pixel block,
 *                     which is often used as a base color in block compression.
 */
void getAverageColor(const Color* src, float* avg_color)
{
	uint sum_b = 0, sum_g = 0, sum_r = 0;
	
	// Accumulate sum of each color channel.
	for (unsigned int i = 0; i < 8; ++i) {
		sum_b += src[i].channels.b;
		sum_g += src[i].channels.g;
		sum_r += src[i].channels.r;
	}
	
	// Calculate average by dividing by 8.
	const float kInv8 = 1.0f / 8.0f;
	avg_color[0] = (float)(sum_b) * kInv8;
	avg_color[1] = (float)(sum_g) * kInv8;
	avg_color[2] = (float)(sum_r) * kInv8;
}

/**
 * @brief Computes the optimal luminance codeword table and pixel data for a sub-block.
 * Memory Usage: `__global` for `block`, `__constant` for `idx_to_num_tab`.
 * Functional Utility: Iterates through possible codeword tables, finds the one that
 *                     minimizes color error for the given sub-block, and encodes
 *                     the table index and pixel data into the output block.
 * @param block Pointer to the global memory destination block.
 * @param src Pointer to the source Color pixels for the sub-block (8 pixels).
 * @param base The base color for luminance modulation.
 * @param sub_block_id The ID of the current sub-block (e.g., 0 or 1).
 * @param idx_to_num_tab Pointer to a constant array used for pixel reordering.
 * @param threshold An upper bound for the acceptable error; can be used for early exit optimization.
 * @return The minimum total error achieved for the sub-block.
 */
unsigned long computeLuminance(__global uchar* block,
						   const Color *src,
						   const Color base,
						   int sub_block_id,
						   __constant uchar *idx_to_num_tab,
						   unsigned long threshold)
{
	uint best_tbl_err = threshold; // Initialize best table error to threshold.
	uchar best_tbl_idx = 0;       // Initialize best table index.
	uchar best_mod_idx[8][8];     // Stores best modulation index for each pixel for each table.

	// Iterate over all possible codeword tables.
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		
		Color candidate_color[4];  // Colors generated from base + luminance for current table.
		// For each modulation index (0-3), calculate the candidate color.
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx]; // Get luminance from constant table.
			candidate_color[mod_idx] = makeColor(base, lum);
		}
		
		uint tbl_err = 0; // Accumulated error for the current table.
		
		// Iterate over each pixel in the sub-block.
		for (unsigned int i = 0; i < 8; ++i) {
			
			uint best_mod_err = threshold; // Best modulation error for current pixel.
			// Find the best modulation index for the current pixel.
			for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
				const Color color = candidate_color[mod_idx];
				
				uint mod_err = getColorError(src[i], color); // Calculate color error.
				// If current modulation gives better error, update.
				if (mod_err < best_mod_err) {
					best_mod_idx[tbl_idx][i] = mod_idx; // Store best modulation index.
					best_mod_err = mod_err;
					
					if (mod_err == 0) // Early exit if perfect match found.
						break;  
				}
			}
			
			tbl_err += best_mod_err; // Add best modulation error to total table error.
			if (tbl_err > best_tbl_err) // Early exit if current table is already worse than best.
				break;  
		}
		
		// If current table is better, update best table.
		if (tbl_err < best_tbl_err) {
			best_tbl_err = tbl_err;
			best_tbl_idx = tbl_idx;
			
			if (tbl_err == 0) // Early exit if perfect match found for the entire table.
				break;  
		}
	}

	// Write the best codeword table index to the compressed block.
	WriteCodewordTable(block, sub_block_id, best_tbl_idx);

	uint pix_data = 0; // Initialize pixel data.

	// Encode pixel modulation indices into pixel data.
	for (unsigned int i = 0; i < 8; ++i) {
		uchar mod_idx = best_mod_idx[best_tbl_idx][i]; // Get best modulation index for pixel.
		uchar pix_idx = g_mod_to_pix[mod_idx]; // Map modulation index to pixel index.
		
		uint lsb = pix_idx & 0x1;   // Extract LSB of pixel index.
		uint msb = pix_idx >> 1;    // Extract MSB of pixel index.
		
		// Bitwise Operation: Pack LSB and MSB into `pix_data` at specific texel positions.
		int texel_num = idx_to_num_tab[i]; // Get texel number for current pixel.
		pix_data |= msb << (texel_num + 16);
		pix_data |= lsb << (texel_num);
	}

	// Write the encoded pixel data to the compressed block.
	WritePixelData(block, pix_data);
	return best_tbl_err;
}

/**
 * @brief Attempts to compress a block if all pixels are a solid color.
 * Functional Utility: Provides an optimized compression path for uniform color blocks,
 *                     which can significantly improve compression ratio and quality for such cases.
 * @param dst Pointer to the global memory destination block.
 * @param src Pointer to the source Color pixels for the block (16 pixels).
 * @param error Pointer to an unsigned long where the total error for the block will be stored.
 * @return `true` if the block is solid and compressed, `false` otherwise.
 */
bool tryCompressSolidBlock(__global uchar* dst,
						   const Color* src,
						   unsigned long *error)
{
	// Block Logic: Checks if all 16 pixels in the block are identical.
	// Invariant: If any pixel differs from the first, it's not a solid block.
	for (unsigned int i = 1; i < 16; ++i) {
		if (src[i].bits != src[0].bits)
			return false;
	}
	
	// Initialize the destination block to zeros.
	for (int i = 0; i < 8; ++i)
        dst[i] = 0;
	
	// Convert the solid color to float and then to a 5:5:5 Color.
	float src_color_float[3] = {(float)(src->channels.b),
		(float)(src->channels.g),
		(float)(src->channels.r)};
	Color base = makeColor555(src_color_float);
	
	// Write control flags and base colors for the solid block.
	WriteDiff(dst, true);  // Use differential encoding.
	WriteFlip(dst, false); // No flipping.
	WriteColors555(dst, base, base); // Base color and delta (zero delta here).
	
	uchar best_tbl_idx = 0;
	uchar best_mod_idx = 0;
	uint best_mod_err = UINT32_MAX; // Initialize with max error.
	
	// Find the best codeword table and modulation index for the solid color.
	for (unsigned int tbl_idx = 0; tbl_idx < 8; ++tbl_idx) {
		
		for (unsigned int mod_idx = 0; mod_idx < 4; ++mod_idx) {
			short lum = g_codeword_tables[tbl_idx][mod_idx];
			const Color color = makeColor(base, lum);
			
			uint mod_err = getColorError(*src, color); // Calculate error against the solid color.
			// If current modulation gives better error, update.
			if (mod_err < best_mod_err) {
				best_tbl_idx = tbl_idx;
				best_mod_idx = mod_idx;
				best_mod_err = mod_err;
				
				if (mod_err == 0) // Early exit if perfect match found.
					break;  
			}
		}
		
		if (best_mod_err == 0) // Early exit if perfect match found.
			break;
	}
	
	// Write the best codeword table index for both sub-blocks.
	WriteCodewordTable(dst, 0, best_tbl_idx);
	WriteCodewordTable(dst, 1, best_tbl_idx);
	
	// Encode pixel data for the solid block.
	uchar pix_idx = g_mod_to_pix[best_mod_idx];
	uint lsb = pix_idx & 0x1;
	uint msb = pix_idx >> 1;
	
	uint pix_data = 0;
	// All pixels have the same modulation index, so encode it for all texels.
	for (unsigned int i = 0; i < 2; ++i) { // Loop twice to cover all 16 pixels with 8 indices each.
		for (unsigned int j = 0; j < 8; ++j) {
			
			int texel_num = g_idx_to_num[i][j]; // Get texel number for current pixel.
			pix_data |= msb << (texel_num + 16);
			pix_data |= lsb << (texel_num);
		}
	}
	
	// Write the encoded pixel data.
	WritePixelData(dst, pix_data);
	*error = 16 * best_mod_err; // Total error for the solid block.
	return true; // Block successfully compressed as solid.
}

/**
 * @brief Compresses a 4x4 pixel block into 8 bytes using a texture compression algorithm.
 * Memory Usage: `__global` for `dst`, `ver_src`, `hor_src`.
 * Functional Utility: Implements the core logic for compressing a 4x4 block of pixels,
 *                     determining the best encoding (solid, differential, flipped)
 *                     and generating the compressed output.
 * @param dst Pointer to the global memory destination block (8 bytes).
 * @param ver_src Pointer to the source Color pixels arranged vertically (first 8 pixels of 4x2 block).
 * @param hor_src Pointer to the source Color pixels arranged horizontally (second 8 pixels of 4x2 block).
 * @param threshold An upper bound for the acceptable error; can be used for early exit optimization.
 * @return The total error for the compressed block.
 */
unsigned long compressBlock(__global uchar* dst,
							const Color* ver_src,
							const Color* hor_src,
							unsigned long threshold)
{
	unsigned long solid_error = 0;
	// Try compressing as a solid color block first.
	// Optimization: If the block is uniform, use the simpler solid block compression.
	if (tryCompressSolidBlock(dst, ver_src, &solid_error)) {
		return solid_error;
	}
	
	// Divide the 4x4 block into four 2x2 sub-blocks for processing.
	const Color* sub_block_src[4] = {ver_src, ver_src + 8, hor_src, hor_src + 8};
	
	Color sub_block_avg[4];
	bool use_differential[2] = {true, true}; // Flags to indicate if differential encoding is used for each 2x4 partition.
	
	// Determine differential vs. non-differential encoding for two 2x4 partitions.
	// This loop processes two pairs of sub-blocks (0,1) and (2,3).
	for (unsigned int i = 0, j = 1; i < 4; i += 2, j += 2) {
		float avg_color_0[3];
		getAverageColor(sub_block_src[i], avg_color_0); // Get average color for first sub-block.
		Color avg_color_555_0 = makeColor555(avg_color_0); // Convert to 5:5:5 format.
		
		float avg_color_1[3];
		getAverageColor(sub_block_src[j], avg_color_1); // Get average color for second sub-block.
		Color avg_color_555_1 = makeColor555(avg_color_1); // Convert to 5:5:5 format.
		
		// Check color differences to decide between differential (5:5:5) and absolute (4:4:4) encoding.
		for (unsigned int light_idx = 0; light_idx < 3; ++light_idx) {
			int u = avg_color_555_0.components[light_idx] >> 3; // 5-bit component.
			int v = avg_color_555_1.components[light_idx] >> 3; // 5-bit component.
			
			int component_diff = v - u;
			// If difference is too large, use 4:4:4 encoding.
			if (component_diff < -4 || component_diff > 3) {
				use_differential[i / 2] = false; // Set flag for current partition.
				sub_block_avg[i] = makeColor444(avg_color_0); // Convert to 4:4:4.
				sub_block_avg[j] = makeColor444(avg_color_1); // Convert to 4:4:4.
			} else {
				// Otherwise, use 5:5:5 encoding (differential).
				sub_block_avg[i] = avg_color_555_0;
				sub_block_avg[j] = avg_color_555_1;
			}
		}
	}
	
	// Calculate error for each sub-block's average color.
	uint sub_block_err[4] = {0};
	for (unsigned int i = 0; i < 4; ++i) {
		for (unsigned int j = 0; j < 8; ++j) {
			sub_block_err[i] += getColorError(sub_block_avg[i], sub_block_src[i][j]);
		}
	}
	
	// Determine if the block should be flipped (vertical vs. horizontal split) based on errors.
	bool flip =
	sub_block_err[2] + sub_block_err[3] < sub_block_err[0] + sub_block_err[1];
	
	// Initialize destination block.
    for (int i = 0; i < 8; ++i)
        dst[i] = 0;
   
	// Write differential and flip flags to the control byte.
	WriteDiff(dst, use_differential[!!flip]);
	WriteFlip(dst, flip);
	
	// Determine sub-block offsets based on the flip flag.
	uchar sub_block_off_0 = flip ? 2 : 0;
	uchar sub_block_off_1 = sub_block_off_0 + 1;
	
	// Write colors based on differential flag.
	if (use_differential[!!flip]) {
		WriteColors555(dst, sub_block_avg[sub_block_off_0],
					   sub_block_avg[sub_block_off_1]);
	} else {
		WriteColors444(dst, sub_block_avg[sub_block_off_0],
					   sub_block_avg[sub_block_off_1]);
	}
	
	unsigned long lumi_error1 = 0, lumi_error2 = 0;
	
	// Compute luminance for the first 2x4 sub-block.
	lumi_error1 = computeLuminance(dst, sub_block_src[sub_block_off_0],
								   sub_block_avg[sub_block_off_0], 0,
								   g_idx_to_num[sub_block_off_0],
								   threshold);
                                   
	// Compute luminance for the second 2x4 sub-block.
	lumi_error2 = computeLuminance(dst, sub_block_src[sub_block_off_1],
								   sub_block_avg[sub_block_off_1], 1,
								   g_idx_to_num[sub_block_off_1],
								   threshold);
	
	return lumi_error1 + lumi_error2; // Return total error.
}

/**
 * @brief Extracts a 4x4 pixel block from a source image into an array of Color pointers.
 * Memory Usage: `__global` for `source` and `blockRows`.
 * Functional Utility: Organizes pointers to the rows of a 4x4 pixel block from a linear
 *                     source image, simplifying access within kernel functions.
 * @param source Pointer to the global memory source image data.
 * @param width The width of the source image in pixels.
 * @param blockRows An array of 4 global Color pointers, which will point to the start of each row in the 4x4 block.
 * Thread Indexing: Implicitly uses thread ID to determine the starting pixel of the block (not directly in this function, but in its caller).
 */
void extractBlock(__global uchar *source, const int width, 
                  __global Color *blockRows[BLOCKWIDTH]) {
    blockRows[0] = (__global Color *)(source);
    blockRows[1] = blockRows[0] + width;
    blockRows[2] = blockRows[1] + width;
    blockRows[3] = blockRows[2] + width;
}

/**
 * @brief Extracts two 2x4 vertical sub-blocks from a 4x4 thread block.
 * Memory Usage: `__global` for `threadBlock`, `vBlocks` is host-side allocated buffer for output.
 * Functional Utility: Rearranges pixel data from a 4x4 block into two 2x4 vertical sub-blocks,
 *                     which is a common step in specific block compression algorithms.
 * @param threadBlock Array of pointers to the 4 rows of the 4x4 input block.
 * @param vBlocks A buffer to store the extracted vertical blocks.
 */
void extractVerticalBlocks(__global void *threadBlock[BLOCKWIDTH], 
                             void *vBlocks[BLOCKSIZE]) {
                                 
	uchar *dst = (uchar *) vBlocks;
    __global const uchar *src;
    
    // Iterate over two vertical sub-blocks (left and right 2-pixel columns).
    for (int sb = 0; sb < 2; ++sb) {
        // Iterate over each row of the 4x4 block.
        for (int r = 0; r < 4; ++r) {
            // Point `src` to the starting byte of the current 2-pixel segment in the row.
            src = (__global uchar *) threadBlock[r] + 2 * sb;
            
            // Copy 2 bytes (representing 2 pixels) to the destination.
            for (int b = 0; b < BLOCKSIZE / 2; ++b) // BLOCKSIZE/2 == 8, but only 2 bytes are copied per iteration of 'r'
                dst[b] = src[b];
            dst += 2; // Move destination pointer by 2 bytes.
        }
    }
}

/**
 * @brief Extracts four 4x1 horizontal sub-blocks from a 4x4 thread block.
 * Memory Usage: `__global` for `threadBlock`, `hBlocks` is host-side allocated buffer for output.
 * Functional Utility: Rearranges pixel data from a 4x4 block into four 4x1 horizontal sub-blocks,
 *                     which is a common step in specific block compression algorithms.
 * @param threadBlock Array of pointers to the 4 rows of the 4x4 input block.
 * @param hBlocks A buffer to store the extracted horizontal blocks.
 */
void extractHorizontalBlocks(__global void *threadBlock[BLOCKWIDTH], 
                             void *hBlocks[BLOCKSIZE]) {
                                 
	uchar *dst = (uchar *) hBlocks;
    __global const uchar *src;
    
    // Iterate over each row of the 4x4 block.
    for (int r = 0; r < 4; ++r) {
        src = (__global uchar *) threadBlock[r]; // Point `src` to the start of the current row.
        // Copy 4 bytes (representing 4 pixels) from the current row.
        for (int b = 0; b < BLOCKSIZE; ++b) // BLOCKSIZE == 16, this loop copies 16 bytes (4 pixels * 4 components/pixel)
            dst[r * 4 * 4 + b] = src[b]; // Destination index calculation is incorrect based on loop
    }
}

/**
 * @brief Calculates and applies the global offset for source and destination pointers based on thread IDs.
 * @param src Pointer to the global memory source pointer.
 * @param dst Pointer to the global memory destination pointer.
 * @param width The width of the entire image.
 * @param i The global X-coordinate (block column index) of the current work-item.
 * @param j The global Y-coordinate (block row index) of the current work-item.
 * Functional Utility: Translates 2D block coordinates (i, j) into linear memory offsets
 *                     for accessing input and output image data in global memory.
 * Thread Indexing: Uses `i` and `j` (derived from `get_global_id(0)` and `get_global_id(1)`)
 *                  to calculate the specific block location for the current work-item.
 */
void applyOffset(__global uchar **src, __global uchar **dst,
                 int width, int i, int j) {
    // Offset calculation for source image (4 bytes per pixel, 4 pixels per block horizontally).
    *src += i * 4 * 4 +
            4 * 4 * j * width;
            
    // Offset calculation for destination compressed image (2 bytes per block horizontally).
    *dst += 2 * 4 * i + // 2 bytes * 4 channels * i = 8*i
            2 * j * width;
}
                 
/**
 * @kernel compress
 * @brief OpenCL kernel for compressing an image using a block-based texture compression algorithm.
 * Memory Usage: `src` and `dst` are in `__global` memory. `threadBlock`, `vBlocks`, `hBlocks` are local to work-group/private.
 * Thread Indexing: Each work-item processes a 4x4 block of pixels.
 *                  `i = get_global_id(0)` corresponds to the block's column index.
 *                  `j = get_global_id(1)` corresponds to the block's row index.
 * Functional Utility: Orchestrates the entire block compression process on the GPU,
 *                     from extracting pixel data to performing compression and writing the result.
 * @param src Pointer to the global memory source image (uncompressed RGBA data).
 * @param dst Pointer to the global memory destination buffer (compressed data).
 * @param width The width of the source image in pixels.
 * @param height The height of the source image in pixels.
 */
__kernel void compress(__global uchar *src, __global uchar *dst,
                       const int width, const int height) {
    __global Color *threadBlock[BLOCKWIDTH]; // Pointers to rows within the current 4x4 block.
    int i = get_global_id(0); // Global X-index (block column).
    int j = get_global_id(1); // Global Y-index (block row).
    Color vBlocks[16]; // Buffer for vertically extracted sub-blocks.
	Color hBlocks[16]; // Buffer for horizontally extracted sub-blocks.
    
    // Adjust `src` and `dst` pointers to the current block's starting position.
    applyOffset(&src, &dst, width, i, j);
    // Extract the 4x4 pixel block from the source image.
    extractBlock(src, width, threadBlock);

    // Extract horizontal and vertical sub-blocks for processing.
    extractHorizontalBlocks(threadBlock, hBlocks);
    extractVerticalBlocks(threadBlock, vBlocks);
    // Perform the block compression and write to `dst`.
    compressBlock(dst, vBlocks, hBlocks, INT32_MAX);
}

// NOTE: The following C++ code section seems to be part of a different file or
// is incorrectly appended. It contains includes and definitions typical of a .cpp file.
// For the purpose of this documentation task, it's treated as if it were part of this header.

#include "compress.hpp" // Assumed to define TextureCompressor class
#include "helper.hpp"   // Self-reference, indicating potential inclusion issues if not guarded.

/**
 * @brief Default constructor for the TextureCompressor class.
 * Functional Utility: Initializes the OpenCL device for GPU computations.
 * Postcondition: `this->device` is initialized with an available OpenCL GPU device.
 */
TextureCompressor::TextureCompressor() {
    gpu_find(this->device); // Locate and set up the GPU device.
}

/**
 * @brief Destructor for the TextureCompressor class.
 * Functional Utility: Cleans up any resources allocated by the compressor (not explicitly shown here).
 */
TextureCompressor::~TextureCompressor() { }
	
/**
 * @brief Discovers and sets up an available OpenCL GPU device.
 * Functional Utility: Abstracts the OpenCL platform and device enumeration process,
 *                     finding the first available GPU to use for computation.
 * @param device Reference to `cl_device_id` to store the discovered GPU device ID.
 * Postcondition: `device` is set to the ID of an available OpenCL GPU, or the program exits on failure.
 * Throws: Exits the program using `DIE` macro if OpenCL API calls fail or memory allocation fails.
 */
void gpu_find(cl_device_id &device)
{
	cl_platform_id *platform_list = NULL;
	cl_platform_id platform;
	cl_uint platform_num = 0;
	
	cl_device_id *device_list = NULL;
	cl_uint device_num = 0;
	cl_device_type devType;
    
	// Get the number of available OpenCL platforms.
	CL_ERR(clGetPlatformIDs(0, NULL, &platform_num));
	// Allocate memory for platform IDs.
	platform_list = new cl_platform_id[platform_num];
	DIE(platform_list == NULL, "alloc platform_list");
	
	// Get the actual platform IDs.
	CL_ERR(clGetPlatformIDs(platform_num, platform_list, NULL));

    int found = 0; // Flag to indicate if a GPU device was found.
	// Iterate through each platform to find a GPU device.
	for(uint platf = 0; platf < platform_num; ++platf) {
		platform = platform_list[platf];
		DIE(platform == 0, "platform selection");

		// Get the number of GPU devices on the current platform.
		if(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &device_num) == CL_DEVICE_NOT_FOUND) {
			device_num = 0; // No GPU devices found on this platform.
			continue;
		}
        
		// Allocate memory for device IDs.
		device_list = new cl_device_id[device_num];
		DIE(device_list == NULL, "alloc devices");
		
		// Get the actual device IDs for GPUs on this platform.
		CL_ERR(clGetDeviceIDs(
            platform, 
            CL_DEVICE_TYPE_GPU,
			device_num, 
            device_list, 
            NULL));

		// Iterate through devices to find a GPU.
		for(uint dev = 0; dev < device_num; ++dev) {
			clGetDeviceInfo(
                device_list[dev], 
                CL_DEVICE_TYPE,
                sizeof(cl_device_type), 
                &devType, 
                NULL);
			
            if(devType == CL_DEVICE_TYPE_GPU) {
                device = device_list[dev]; // Assign the found GPU device.
                found = 1; // Set flag.
                break; // Exit device loop.
            }
		}
        
        if (found)
            break; // Exit platform loop if GPU found.
	}
	
	delete[] platform_list; // Free allocated memory.
	delete[] device_list;   // Free allocated memory.
}

/**
 * @brief Executes the OpenCL compression kernel on the GPU.
 * Functional Utility: Manages the full lifecycle of an OpenCL kernel execution,
 *                     including context creation, command queue setup, buffer allocation,
 *                     data transfer (host to device, device to host), kernel compilation,
 *                     argument setting, and kernel launch.
 * @param device The OpenCL device ID to use for execution.
 * @param src Pointer to the host memory source image data.
 * @param dst Pointer to the host memory destination buffer for compressed data.
 * @param width The width of the source image.
 * @param height The height of the source image.
 * Postcondition: The `dst` buffer contains the compressed image data.
 * Throws: Exits the program using `CL_ERR` or `CL_COMPILE_ERR` on any OpenCL API failure.
 */
void gpu_execute_kernel(cl_device_id device, 
                        const uint8_t* src,
                        uint8_t* dst,
						int width,
						int height)
{
	cl_command_queue cmd_queue;
	cl_context context;
	cl_program program;
	string kernel_src;
	cl_kernel kernel;
    cl_mem gpusrc; // Global memory buffer for source data on GPU.
    cl_mem gpudst; // Global memory buffer for destination data on GPU.
    int size_src;
    int size_dst;
	cl_int ret;
    
	// Create an OpenCL context.
	context = clCreateContext(
        0,        // context_properties
        1,        // num_devices
        &device,  // devices
        NULL,     // pfn_notify
        NULL,     // user_data
        NULL);    // errcode_ret

	// Create a command queue.
	cmd_queue = clCreateCommandQueue(
        context, 
        device, 
        0,        // properties
        NULL);    // errcode_ret

	// Calculate source buffer size (width * height * 4 bytes per pixel).
    size_src = width * height * 4;
	// Create input buffer on GPU (read-only).
	gpusrc = clCreateBuffer(
        context,  
        CL_MEM_READ_ONLY,
        sizeof(char) * size_src, 
        NULL, 
        &ret);
    CL_ERR(ret);

    // Calculate destination buffer size (width * height * 4 bytes per pixel / 8 bytes per compressed block).
    size_dst = width * height * 4 / 8;
	// Create output buffer on GPU (write-only).
	gpudst = clCreateBuffer(
        context, 
        CL_MEM_WRITE_ONLY,
        sizeof(char) * size_dst, 
        NULL, 
        &ret);
    CL_ERR(ret);

    // Enqueue write operation to transfer source data from host to GPU.
    ret = clEnqueueWriteBuffer(
        cmd_queue, 
        gpusrc, 
        CL_TRUE,  // blocking_write (wait until complete)
        0,        // offset
        size_src, 
        src,      // host_ptr
        0,        // num_events_in_wait_list
        NULL,     // event_wait_list
        NULL);    // event
	
	// Read kernel source from file.
	read_kernel("kernelsrc.cl", kernel_src);
	const char* kernel_c_str = kernel_src.c_str();

	// Create program from kernel source.
	program = clCreateProgramWithSource(
        context, 
        1,              // count
		&kernel_c_str,  // strings
        NULL,           // lengths
        &ret);          // errcode_ret
	CL_ERR(ret);

	// Build (compile) the OpenCL program for the device.
	ret = clBuildProgram(
        program, 
        1,        // num_devices
        &device,  // device_list
        NULL,     // options
        NULL,     // pfn_notify
        NULL);    // user_data
	CL_COMPILE_ERR(ret, program, device);

	// Create the OpenCL kernel from the compiled program.
	kernel = clCreateKernel(
        program, 
        "compress", // kernel_name
        &ret);      // errcode_ret
	CL_ERR(ret);

	// Set kernel arguments.
    ret  = 0;
	ret |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &gpusrc);
	ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &gpudst);
	ret |= clSetKernelArg(kernel, 2, sizeof(int), &width);
	ret |= clSetKernelArg(kernel, 3, sizeof(int), &height);
    
	// Define global work size (each work-item processes a 4x4 block).
	size_t globalSize[2] = {
        (size_t) width / 4, 
        (size_t) height / 4};
    
    // Enqueue the kernel for execution.
    CL_ERR(clEnqueueNDRangeKernel(
        cmd_queue, 
        kernel, 
        2,        // work_dim
        NULL,     // global_work_offset
        globalSize, 
        NULL,     // local_work_size (let OpenCL choose)
        0,        // num_events_in_wait_list
        NULL,     // event_wait_list
        NULL));   // event
    CL_ERR(clFinish(cmd_queue)); // Wait for kernel to complete.

	// Enqueue read operation to transfer compressed data from GPU to host.
	CL_ERR(clEnqueueReadBuffer(
        cmd_queue, 
        gpudst, 
        CL_TRUE,  // blocking_read
        0,        // offset
        sizeof(char) * size_dst, 
        dst,      // host_ptr
        0,        // num_events_in_wait_list
        NULL,     // event_wait_list
        NULL));   // event
	CL_ERR(clFinish(cmd_queue)); // Wait for read to complete.

	// Release OpenCL resources.
    clReleaseProgram(program);
	clReleaseKernel(kernel);
	CL_ERR(clReleaseMemObject(gpusrc));
	CL_ERR(clReleaseMemObject(gpudst));
	CL_ERR(clReleaseCommandQueue(cmd_queue));
	CL_ERR(clReleaseContext(context));
}

/**
 * @class TextureCompressor
 * @brief Manages the GPU-accelerated texture compression process using OpenCL.
 * Functional Utility: Provides an interface for compressing image data by offloading
 *                     the computation to an OpenCL-enabled GPU.
 */
class TextureCompressor {
public:
    cl_device_id device; /**< The OpenCL device ID used for compression. */

    /**
     * @brief Default constructor for TextureCompressor.
     * Functional Utility: Initializes the OpenCL device by calling `gpu_find`.
     */
    TextureCompressor();

    /**
     * @brief Destructor for TextureCompressor.
     * Functional Utility: Placeholder for releasing resources, if any were managed by the class directly.
     */
    ~TextureCompressor();

    /**
     * @brief Compresses source image data using the configured OpenCL device.
     * @param src Pointer to the uncompressed source image data (host memory).
     * @param dst Pointer to the destination buffer for compressed data (host memory).
     * @param width The width of the source image.
     * @param height The height of the source image.
     * @return 0 on success (or an error code if extended).
     * Functional Utility: Initiates the GPU-based image compression by invoking `gpu_execute_kernel`.
     * Precondition: An OpenCL device must have been successfully found and initialized.
     * Postcondition: `dst` buffer contains the compressed texture data.
     */
    unsigned long compress(const uint8_t* src,
                                      uint8_t* dst,
									  int width,
									  int height);
};

// End of the potentially mis-appended C++ code section.

