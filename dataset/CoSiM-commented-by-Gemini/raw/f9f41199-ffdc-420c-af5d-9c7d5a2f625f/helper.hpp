/**
 * @file helper.hpp
 * @brief Provides utility functions for OpenCL error handling and kernel loading.
 *
 * This file contains a set of helper functions designed to simplify OpenCL
 * programming by providing robust error checking, meaningful error messages, and
 * an easy way to load kernel source from a file.
 */
#ifndef CL_HELPER_H
#define CL_HELPER_H

#if __APPLE__
   #include <OpenCL/opencl.h>
#else
   #include <CL/cl.h>
#endif

#include <string>

using namespace std;

/**
 * @brief Checks for OpenCL runtime errors.
 * @param cl_ret The return value from an OpenCL API call.
 * @return The same error code, for chaining.
 */
int CL_ERR(int cl_ret);

/**
 * @brief Checks for OpenCL program compilation errors.
 * @param cl_ret The return value from clBuildProgram.
 * @param program The OpenCL program object.
 * @param device The device on which the program was compiled.
 * @return The same error code.
 */
int CL_COMPILE_ERR(int cl_ret,
                  cl_program program,
                  cl_device_id device);

/**
 * @brief Converts an OpenCL error code to a human-readable string.
 * @param err The OpenCL error code.
 * @return A C-style string describing the error.
 */
const char* cl_get_string_err(cl_int err);

/**
 * @brief Retrieves and prints the compiler error log for a given program and device.
 * @param program The OpenCL program object.
 * @param device The device for which the build log is queried.
 */
void cl_get_compiler_err_log(cl_program program,
                             cl_device_id device);

/**
 * @brief Reads the content of a kernel file into a string.
 * @param file_name The path to the kernel source file.
 * @param str_kernel A reference to a string where the file content will be stored.
 */
void read_kernel(string file_name, string &str_kernel);

/**
 * @def DIE(assertion, call_description)
 * @brief A macro for fatal error checking. If the assertion is true, it prints
 *        an error message along with the file name and line number, and then
 *        terminates the program.
 *
 * This macro is a convenient way to handle fatal errors, reducing boilerplate code.
 */
#define DIE(assertion, call_description)                    \
do {                                                        \
    if (assertion) {                                        \
            fprintf(stderr, "(%d): ",                       \
                            __LINE__);                      \
            perror(call_description);
            exit(EXIT_FAILURE);
    }
} while(0);

#endif
