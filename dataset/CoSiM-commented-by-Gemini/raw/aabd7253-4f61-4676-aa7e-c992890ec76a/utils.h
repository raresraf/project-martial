/**
 * @file utils.h
 * @brief Utility functions and data structures for testing matrix solvers.
 *
 * This header file defines common structures, macros, and function prototypes
 * used across different solver implementations and their testing harnesses. It
 * provides a standardized way to handle test data generation, execution, and
 * I/O.
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

/**
 * @brief A function pointer type representing a matrix solver.
 *
 * A Solver is expected to take the matrix size N, a pointer to the matrix A,
 * and a pointer to the vector B, and return a pointer to the solution vector x
 * for the system Ax = B.
 */
typedef double* (*Solver)(int, double *, double*);

/**
 * @brief Generates a random double-precision floating-point number within a given range.
 * @param limit The positive limit of the range. The generated number will be in [-limit, limit].
 */
#define get_rand_double(limit) ((((double)rand()) / RAND_MAX) * (2 * limit) - limit)

/**
 * @struct test
 * @brief A structure to hold the parameters for a single solver test case.
 */
struct test {
	/** @brief The seed for the random number generator, ensuring reproducibility. */
	int seed;
	/** @brief The size (dimension) of the square matrix N x N. */
	int N;
	
	/** @brief The path to the file where the output/results should be saved. */
	char output_save_file[100];
};

/**
 * @brief The signature for the solver function to be implemented.
 */
double* my_solver(int, double *, double *);

/**
 * @brief Runs a single test case for a given solver.
 * @param t The test parameters.
 * @param solver A function pointer to the solver implementation to be tested.
 * @param elapsed_time A pointer to a float where the execution time will be stored.
 * @return An integer status code (typically 0 for success).
 */
int run_test(struct test, Solver, float *);

/**
 * @brief Frees the memory allocated for the test data (matrix and vectors).
 * @param data A pointer to the array of allocated data pointers.
 */
void free_data(double **);

/**
 * @brief Generates the random matrix A and vector B for a test case.
 * @param t The test parameters, including the matrix size N and random seed.
 * @param data A pointer to an array where pointers to the generated matrix and vector will be stored.
 * @param no_data The number of data arrays to generate.
 * @return An integer status code (typically 0 for success).
 */
int generate_data(struct test, double **, int);

/**
 * @brief Reads test case configurations from an input file.
 * @param input_file The path to the input configuration file.
 * @param no_tests A pointer to an integer where the number of tests read will be stored.
 * @param tests A pointer to an array of `test` structures that will be allocated and filled.
 * @return An integer status code (typically 0 for success).
 */
int read_input_file(char *, int *, struct test **);
