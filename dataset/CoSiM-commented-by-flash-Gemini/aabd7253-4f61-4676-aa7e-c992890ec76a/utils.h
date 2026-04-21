
/**
 * @file utils.h
 * @brief Common utilities and interface definitions for matrix solver implementations.
 * 
 * Functional Intent: Provides a unified framework for testing and validating 
 * matrix multiplication solvers. It defines the generic `Solver` function 
 * pointer type, data generation macros, and structures for orchestrating 
 * performance benchmarks with reproducible seeds and varied matrix sizes.
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

/**
 * @typedef Solver
 * @brief Standardized signature for matrix solver functions.
 * 
 * @param int N - Matrix dimension (NxN).
 * @param double* A - Input matrix A.
 * @param double* B - Input matrix B.
 * @return double* - Resulting matrix.
 */
typedef double* (*Solver)(int, double *, double*);

/**
 * @def get_rand_double
 * @brief Generates a random double within the range [-limit, limit].
 */
#define get_rand_double(limit) ((((double)rand()) / RAND_MAX) * (2 * limit) - limit)

/**
 * @struct test
 * @brief Configuration for a single solver performance test.
 */
struct test {
	int seed; // Reproducibility seed for PRNG.
	int N;    // Matrix size.
	
	char output_save_file[100]; // Target file for result validation.
};

/**
 * my_solver - Declaration for the variant-specific solver implementation.
 */
double* my_solver(int, double *, double *);

/**
 * @brief Executes a test instance and measures wall-clock execution time.
 */
int run_test(struct test, Solver, float *);

void free_data(double **);

/**
 * @brief Populates memory buffers with randomized floating-point data.
 */
int generate_data(struct test, double **, int);

/**
 * @brief Parses a configuration file to extract an array of test scenarios.
 */
int read_input_file(char *, int *, struct test **);
