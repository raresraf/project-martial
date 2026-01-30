/**
 * @file helper.cpp
 * @brief Implementation of helper functions for OpenCL integration.
 *
 * This file provides utility functions for error handling, kernel compilation,
 * and file I/O, simplifying the interaction with the OpenCL API. It is a part
 * of a texture compression project, providing common functionalities needed
 * to set up and run OpenCL kernels.
 *
 * Architectural Intent:
 * - To encapsulate common OpenCL boilerplate code into reusable functions.
 * - To provide a centralized and consistent way of handling OpenCL errors and
 *   compiler logs.
 * - To abstract file reading operations for loading OpenCL kernel source code.
 */
#include 
#include 
#include 
#include 
#include 

#include "helper.hpp"

using namespace std;

/**
 * @brief Checks the return code of an OpenCL function and prints an error
 *        message if it's not CL_SUCCESS.
 * @param cl_ret The return code from the OpenCL function.
 * @return 1 if an error occurred, 0 otherwise.
 *
 * This function serves as a simple error-checking wrapper for OpenCL API calls.
 * If the return code indicates an error, it prints the corresponding error string
 * to standard output.
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
 * @brief Checks the return code of an OpenCL program compilation and prints
 *        compiler error logs if compilation fails.
 * @param cl_ret The return code from the compilation function (e.g., `clBuildProgram`).
 * @param program The OpenCL program object that was compiled.
 * @param device The device for which the program was compiled.
 * @return 1 if a compilation error occurred, 0 otherwise.
 *
 * This function extends `CL_ERR` by also fetching and printing the compiler's
 * build log in case of a `CL_BUILD_PROGRAM_FAILURE`, which is crucial for
 * debugging OpenCL kernel code.
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
 * @brief Reads the source code of an OpenCL kernel from a file into a string.
 * @param file_name The path to the kernel file.
 * @param[out] str_kernel A reference to a string where the kernel source will be stored.
 *
 * This function opens a file, reads its entire content into a stringstream,
 * and then populates the output string `str_kernel`. It includes an assertion
 * to ensure the file was opened successfully.
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
 * @param err The OpenCL error code (`cl_int`).
 * @return A constant character pointer to a string describing the error.
 *
 * This function provides a mapping from OpenCL's numeric error codes to
 * descriptive strings, which is essential for meaningful error reporting.
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
 * @brief Retrieves and prints the build log for an OpenCL program.
 * @param program The OpenCL program object.
 * @param device The device for which the program was compiled.
 *
 * This function is called when a program compilation fails. It queries the
 * build log from the OpenCL runtime and prints it to standard output, which
 * is essential for debugging kernel syntax and compilation errors.
 */
void cl_get_compiler_err_log(cl_program program, cl_device_id device)
{
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