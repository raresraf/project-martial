/**
 * @file helper.cpp
 * @brief Implements utility functions for OpenCL programming, including error
 *        checking, kernel source loading, and compiler log retrieval.
 *        These functions facilitate robust OpenCL application development.
 */

#include <iostream>  // For standard input/output operations (cout, endl).
#include <fstream>   // For file stream operations (ifstream).
#include <sstream>   // For string stream operations (stringstream).
#include <string>    // For string manipulation (std::string).
#include <cstdio>    // For perror.
#include <cstdlib>   // For exit.
#include <CL/cl.h>   // OpenCL API.

#include "helper.hpp" // Custom helper header.

using namespace std; // Using the standard namespace for convenience.

/**
 * @brief Checks the return code of an OpenCL API call.
 *
 * This function evaluates the provided OpenCL return code (`cl_ret`).
 * If the code indicates an error (not `CL_SUCCESS`), it prints a descriptive
 * error message to standard output using `cl_get_string_err` and returns 1.
 * Otherwise, it returns 0. This is typically used after OpenCL function calls
 * that return `cl_int`.
 *
 * @param cl_ret The integer return code from an OpenCL API function.
 * @return int Returns 1 if `cl_ret` is an error, 0 if `CL_SUCCESS`.
 */
int CL_ERR(int cl_ret)
{
	// Pre-condition: cl_ret contains the status of an OpenCL operation.
	if(cl_ret != CL_SUCCESS){
		// If an error occurred, print the error string and indicate failure.
		cout << endl << cl_get_string_err(cl_ret) << endl;
		// Invariant: The function will return 1, signifying an error.
		return 1;
	}
	// Invariant: The function will return 0, signifying success.
	return 0;
}

/**
 * @brief Checks the return code from an OpenCL program compilation.
 *
 * Similar to `CL_ERR`, but specialized for `clBuildProgram`'s return code.
 * If `cl_ret` indicates a compilation failure, it prints the generic error
 * and then calls `cl_get_compiler_err_log` to display the detailed build log
 * from the OpenCL device. This is crucial for diagnosing kernel compilation issues.
 *
 * @param cl_ret The integer return code from `clBuildProgram`.
 * @param program The `cl_program` object that was compiled.
 * @param device The `cl_device_id` on which the program was built.
 * @return int Returns 1 if a compilation error occurred, 0 otherwise.
 */
int CL_COMPILE_ERR(int cl_ret, cl_program program, cl_device_id device)
{
	// Pre-condition: cl_ret contains the status of an OpenCL program build.
	if(cl_ret != CL_SUCCESS){
		// If a compilation error occurred, print the error and the build log.
		cout << endl << cl_get_string_err(cl_ret) << endl;
		cl_get_compiler_err_log(program, device);
		// Invariant: The function will return 1, signifying a compilation error.
		return 1;
	}
	// Invariant: The function will return 0, signifying successful compilation.
	return 0;
}

/**
* @brief Reads the content of an OpenCL kernel file into a string.
*
* This function attempts to open the file specified by `file_name`.
* If successful, it reads the entire content of the file and stores
* it in the `str_kernel` reference. If the file cannot be opened,
* it uses the `DIE` macro to terminate the program with an error message.
*
* @param file_name The path to the OpenCL kernel source file.
* @param str_kernel A reference to a string where the kernel source will be stored.
*/
void read_kernel(string file_name, string &str_kernel)
{
	// Attempt to open the file.
	ifstream in_file(file_name.c_str());
	// Pre-condition: The file must be successfully opened.
	DIE( !in_file.is_open(), "ERR OpenCL kernel file. Same directory as binary ?" );

	// Read the file's content into a stringstream.
	stringstream str_stream;
	str_stream << in_file.rdbuf();

	// Assign the stringstream's content to the output string.
	str_kernel = str_stream.str();
	// Invariant: str_kernel now contains the full content of the file.
}

/**
 * @brief Returns a human-readable string for an OpenCL error code.
 *
 * This function implements a switch-case statement to map various `cl_int`
 * error codes, as defined by the OpenCL specification, to their corresponding
 * descriptive string literals. It covers common API errors, memory management
 * issues, compilation failures, and invalid argument errors.
 *
 * @param err The `cl_int` error code returned by an OpenCL API function.
 * @return const char* A C-style string literal describing the error.
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
 * @brief Retrieves and prints the OpenCL program build log.
 *
 * This function queries the OpenCL runtime for the build log associated with
 * a specific program and device. It first determines the required buffer size
 * for the log, then allocates memory, retrieves the log, and prints it to
 * standard output. This is vital for debugging compilation errors in OpenCL kernels.
 *
 * @param program The `cl_program` object for which to retrieve the build log.
 * @param device The `cl_device_id` from which the program was built.
 */
void cl_get_compiler_err_log(cl_program program, cl_device_id device)
{
	char* build_log;
	size_t log_size;

	/* first call to know the proper size */
	// Pre-condition: program and device are valid OpenCL handles.
	// Invariant: log_size will contain the size of the build log.
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
						  0, NULL, &log_size);
	// Allocate memory for the build log, including space for null terminator.
	build_log = new char[ log_size + 1 ];

	/* second call to get the log */
	// Invariant: build_log will be populated with the compiler's output.
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
						  log_size, build_log, NULL);
	// Null-terminate the string.
	build_log[ log_size ] = '\0';
	// Print the build log to console.
	cout << endl << build_log << endl;
	// Clean up allocated memory.
	delete[] build_log;
}
