
/**
 * @file utils.h
 * @brief Provides utility functions, type definitions, and macros for a matrix solver application.
 *        This file includes functionalities for generating random numbers, defining solver interfaces,
 *        handling test configurations, and managing data for matrix operations.
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

/**
 * @brief Type definition for a solver function.
 *
 * A `Solver` is a function pointer that takes the matrix dimension (int N) and
 * two pointers to double-precision matrices (double *A, double *B) as input,
 * and returns a pointer to a double-precision matrix (double*) as output.
 */
typedef double* (*Solver)(int, double *, double*);

/**
 * @brief Generates a random double-precision floating-point number within a specified range.
 *
 * @param limit The absolute maximum value for the random number.
 *              The generated number will be between -limit and +limit.
 */
#define get_rand_double(limit) ((((double)rand()) / RAND_MAX) * (2 * limit) - limit)

/**
 * @brief Structure to hold test case configuration.
 *
 * This structure encapsulates parameters required for running a single test case,
 * including a seed for random number generation, matrix dimension, and file paths.
 */
struct test {
	// seed: Seed for the random number generator, ensuring reproducible test data.
	int seed;
	// N: Dimension of the square matrices (N x N) used in the test.
	int N;
	
	// output_save_file: Character array to store the path where output results should be saved.
	char output_save_file[100];
};

/**
 * @brief Placeholder for the actual solver implementation.
 *
 * This function is expected to implement the core matrix solving logic.
 *
 * @param N The dimension of the square matrices.
 * @param A Pointer to the first input matrix.
 * @param B Pointer to the second input matrix.
 * @return Pointer to the resulting matrix after computation.
 */
double* my_solver(int, double *, double *);

/**
 * @brief Runs a single test case with a given solver.
 *
 * Executes a test defined by the `test` struct using the provided `Solver` function
 * and measures its performance.
 *
 * @param test The test configuration structure.
 * @param solver A function pointer to the solver implementation to be tested.
 * @param time Pointer to a float where the execution time of the solver will be stored.
 * @return 0 on success, -1 on failure.
 */
int run_test(struct test, Solver, float *);

/**
 * @brief Frees memory allocated for matrix data.
 *
 * @param data A pointer to a pointer to the double-precision matrix to be freed.
 */
void free_data(double **);

/**
 * @brief Generates random matrix data for a test case.
 *
 * Fills matrices A and B with random double-precision floating-point numbers based on
 * the provided test configuration.
 *
 * @param test The test configuration structure.
 * @param data A pointer to a pointer to the data array to be populated.
 * @param is_input Flag indicating if the data is for input matrices (A and B).
 * @return 0 on success, -1 on memory allocation failure.
 */
int generate_data(struct test, double **, int);

/**
 * @brief Reads test configurations from an input file.
 *
 * Parses a file to extract multiple test case configurations.
 *
 * @param file_name Path to the input file containing test configurations.
 * @param tests_no Pointer to an integer where the number of tests found will be stored.
 * @param tests Pointer to a pointer to an array of `struct test` to store the read configurations.
 * @return 0 on success, -1 on file opening or parsing error.
 */
int read_input_file(char *, int *, struct test **);
