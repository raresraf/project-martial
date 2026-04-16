/**
 * @file utils.h
 * @brief Utility functions and data structures for a matrix solver benchmark.
 *
 * This header file defines the common data structures (`test`), function prototypes,
 * and macros used across different solver implementations and the main test driver.
 * It provides a standardized interface for generating test data, running solver
 * benchmarks, and managing memory.
 */
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

/**
 * @brief A function pointer type definition for a solver function.
 *
 * Any function that conforms to this signature can be benchmarked by the test harness.
 * It is expected to take the matrix size (N), a matrix A, and a matrix B as input,
 * and return a pointer to the resulting solution matrix.
 *
 * @param N The size of the N x N matrices.
 * @param A A pointer to the first input matrix (N x N).
 * @param B A pointer to the second input matrix (N x N).
 * @return A pointer to the resulting N x N matrix.
 */
typedef double* (*Solver)(int, double *, double*);

/**
 * @brief Macro to generate a random double within a given limit.
 * @param limit The upper and lower bound for the random number (e.g., a limit of 10.0
 *              will produce a number between -10.0 and 10.0).
 */
#define get_rand_double(limit) ((((double)rand()) / RAND_MAX) * (2 * limit) - limit)

/**
 * @struct test
 * @brief A structure to hold the parameters for a single benchmark test run.
 */
struct test {
	int seed;                   /**< The seed for the random number generator. */
	int N;                      /**< The dimension of the square matrices (N x N). */
	
	char output_save_file[100]; /**< The file path to save the output, if any. */
};

/**
 * @brief Prototype for the solver function to be implemented.
 */
double* my_solver(int, double *, double *);

/**
 * @brief Prototype for the test execution function.
 *
 * This function orchestrates a single test run for a given solver.
 *
 * @param t The test parameters.
 * @param solver A function pointer to the solver implementation to be tested.
 * @param elapsed_time A pointer to a float where the execution time will be stored.
 * @return An integer status code (0 for success).
 */
int run_test(struct test, Solver, float *);

/**
 * @brief Prototype for the memory deallocation function.
 * @param data A pointer to the array of matrix pointers to be freed.
 */
void free_data(double **);

/**
 * @brief Prototype for the data generation function.
 *
 * This function allocates and initializes the input matrices for a test run.
 *
 * @param t The test parameters, including the matrix size N.
 * @param data A pointer to an array that will hold the pointers to the generated matrices.
 * @param num_matrices The number of matrices to generate.
 * @return An integer status code (0 for success).
 */
int generate_data(struct test, double **, int);

/**
 * @brief Prototype for the function that reads test configurations from a file.
 *
 * @param file_name The path to the input configuration file.
 * @param num_tests A pointer to an integer where the number of tests will be stored.
 * @param tests A pointer to an array of `test` structs that will be allocated and filled.
 * @return An integer status code (0 for success).
 */
int read_input_file(char *, int *, struct test **);
