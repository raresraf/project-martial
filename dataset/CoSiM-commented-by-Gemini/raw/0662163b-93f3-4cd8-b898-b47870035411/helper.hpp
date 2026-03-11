
/**
 * @file helper.hpp
 * @brief Header file for OpenCL helper functions.
 *
 * This file declares a set of utility functions and a macro for simplifying OpenCL host code.
 * These include functions for error checking, reading kernel files, and a macro for handling
 * fatal errors.
 */
#ifndef CL_HELPER_H
#define CL_HELPER_H

#if __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

using namespace std;

/**
 * @brief Checks the return code of an OpenCL function and prints an error message if it's not CL_SUCCESS.
 * @param cl_ret The return code from an OpenCL function.
 * @return 1 if there was an error, 0 otherwise.
 */
int CL_ERR(int cl_ret);

/**
 * @brief Checks the return code of an OpenCL program compilation and prints error logs if compilation failed.
 * @param cl_ret The return code from clBuildProgram.
 * @param program The OpenCL program object.
 * @param device The device for which the program was compiled.
 * @return 1 if there was a compilation error, 0 otherwise.
 */
int CL_COMPILE_ERR(int cl_ret, cl_program program, cl_device_id device);

/**
 * @brief Converts an OpenCL error code into a human-readable string.
 * @param err The OpenCL error code.
 * @return A const char* pointing to the error message string.
 */
const char* cl_get_string_err(cl_int err);

/**
 * @brief Retrieves and prints the compilation log for a failed OpenCL program build.
 * @param program The OpenCL program object that failed to build.
 * @param device The device for which the program was being compiled.
 */
void cl_get_compiler_err_log(cl_program program, cl_device_id device);

/**
* @brief Reads an OpenCL kernel source file into a string.
* @param file_name The name of the kernel file.
* @param str_kernel A reference to a string where the kernel source will be stored.
*/
void read_kernel(string file_name, string &str_kernel);

/**
 * @def DIE(assertion, call_description)
 * @brief A macro that checks an assertion and, if it is true, prints an error message
 *        and exits the program. This is useful for handling unrecoverable errors.
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
