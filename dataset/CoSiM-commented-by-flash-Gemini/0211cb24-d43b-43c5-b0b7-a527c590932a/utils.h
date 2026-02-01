
/**
 * @file utils.h
 * @brief Provides utility functions, type definitions, and macros for testing and benchmarking matrix solver implementations.
 * This header defines a common interface for solver functions, structures for test configurations,
 * and helper functions for data generation, input reading, and memory management.
 * Architectural Intent: To abstract common functionalities required by different matrix solver
 * implementations and their testing harness, promoting code reuse and modularity across various
 * optimized and non-optimized solver versions.
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

typedef double* (*Solver)(int, double *, double*);

/**
 * @brief Generates a random double-precision floating-point number within a specified limit.
 * The generated number will be in the range [-limit, limit].
 * @param limit The maximum absolute value of the random number.
 * @return A random double within the specified range.
 */
#define get_rand_double(limit) ((((double)rand()) / RAND_MAX) * (2 * limit) - limit)

/**
 * @brief Structure to define parameters for a test case.
 * It includes seed for random number generation, matrix dimension N,
 * and the file path to save the output.
 */
struct test {
	int seed;
	int N;
	/**
	 * @brief File path for saving test output.
	 * This field stores the name of the file where the results of a test run should be saved,
	 * allowing for verification against expected outputs.
	 */
	char output_save_file[100];
};

/**
 * @brief Declaration of the solver function to be implemented by various solver files.
 * This function is expected to take matrix dimensions and input matrices, and return
 * a dynamically allocated result matrix.
 * @param N The dimension of the square matrices.
 * @param A The first input matrix (N x N).
 * @param B The second input matrix (N x N).
 * @return A pointer to the newly allocated result matrix (N x N).
 */
double* my_solver(int, double *, double *);

/**
 * @brief Executes a single test case using a provided solver function.
 * This function likely sets up the test environment, generates data, calls the solver,
 * measures performance (if 'float *' is for timing), and potentially verifies results.
 * @param test The test configuration structure.
 * @param Solver The function pointer to the solver implementation to be tested.
 * @param execution_time A pointer to a float where the execution time of the solver will be stored.
 * @return An integer representing the test result (e.g., 0 for success, non-zero for failure).
 */
int run_test(struct test, Solver, float *);

/**
 * @brief Frees dynamically allocated matrix data.
 * @param data A pointer to a pointer to the double array representing the matrix.
 */
void free_data(double **);

/**
 * @brief Generates random test data (matrices) based on the test configuration.
 * @param test The test configuration structure.
 * @param data A pointer to a pointer where the generated matrix data will be stored.
 * @param offset The starting index for data generation (if multiple matrices are generated).
 * @return An integer representing the success or failure of data generation.
 */
int generate_data(struct test, double **, int);

/**
 * @brief Reads test configurations from an input file.
 * The input file is expected to contain parameters for multiple test cases.
 * @param file_name The path to the input file.
 * @param num_tests A pointer to an integer where the number of tests read will be stored.
 * @param tests A pointer to a pointer to an array of 'struct test' where the configurations will be stored.
 * @return An integer representing the success or failure of reading the input file.
 */
int read_input_file(char *, int *, struct test **);
