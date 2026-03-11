
/**
 * @file utils.h
 * @brief A header file containing common definitions and function prototypes for the matrix solvers.
 *
 * This file defines the common data structures, a function pointer type for the solvers,
 * and prototypes for utility functions used across the different solver implementations
 * and testing utilities.
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

/**
 * @brief A function pointer type for the matrix solver functions.
 *
 * This allows different solver implementations (e.g., BLAS, optimized, non-optimized)
 * to be used interchangeably by the testing framework.
 */
typedef double* (*Solver)(int, double *, double*);

/**
 * @def get_rand_double(limit)
 * @brief A macro to generate a random double-precision number within a given range.
 */
#define get_rand_double(limit) ((((double)rand()) / RAND_MAX) * (2 * limit) - limit)

/**
 * @struct test
 * @brief A struct to hold the parameters for a single test case.
 */
struct test {
	int seed; ///< The seed for the random number generator.
	int N;    ///< The size of the square matrices.
	
	char output_save_file[100]; ///< The path to the file where the output should be saved.
};

/**
 * @brief The prototype for the matrix solver function.
 */
double* my_solver(int, double *, double *);

/**
 * @brief Runs a single test case with a given solver.
 * @param t The test parameters.
 * @param solve The solver function to use.
 * @param elapsed_time A pointer to a float where the elapsed time will be stored.
 * @return 0 on success, non-zero on failure.
 */
int run_test(struct test, Solver, float *);

/**
 * @brief Frees the memory allocated for the matrices.
 * @param data A pointer to an array of matrix pointers.
 */
void free_data(double **);

/**
 * @brief Generates the input matrices for a test case.
 * @param t The test parameters.
 * @param data A pointer to an array where the matrix pointers will be stored.
 * @param num_matrices The number of matrices to generate.
 * @return 0 on success, non-zero on failure.
 */
int generate_data(struct test, double **, int);

/**
 * @brief Reads the test parameters from an input file.
 * @param file_path The path to the input file.
 * @param num_tests A pointer to an integer where the number of tests will be stored.
 * @param tests A pointer to an array of test structs where the test parameters will be stored.
 * @return 0 on success, non-zero on failure.
 */
int read_input_file(char *, int *, struct test **);
