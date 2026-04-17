/**
 * @file utils.h
 * @brief Header file defining utility functions and data structures for a matrix solver test framework.
 * @details This file provides the core definitions for test orchestration, including a solver function
 * pointer, a macro for random data generation, and prototypes for test execution, data management, and
 * input parsing. It is central to a framework that compares different matrix solver implementations.
 */

// @raw/22305609-127e-41e5-9a4e-c0e3f35fcc25/utils.h
// Module Purpose: Utility definitions and declarations for solvers.

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

/**
 * @brief Function pointer definition for a matrix solver.
 * @param N The dimension of the square matrices.
 * @param A A pointer to the first input matrix (N x N).
 * @param B A pointer to the second input matrix (N x N).
 * @return A pointer to the resulting matrix C (N x N).
 *
 * This typedef establishes a common interface for all solver implementations,
 * allowing them to be used interchangeably within the test framework.
 */
typedef double* (*Solver)(int, double *, double*);

/**
 * @def get_rand_double(limit)
 * @brief Generates a random double-precision floating-point number within a symmetric range.
 * @param limit The absolute boundary for the random number generation. The generated number
 *              will be in the range [-limit, limit].
 *
 * This macro is used for populating matrices with random test data.
 */
#define get_rand_double(limit) ((((double)rand()) / RAND_MAX) * (2 * limit) - limit)

/**
 * @struct test
 * @brief Defines the parameters for a single test case.
 * @var test::seed
 * An integer seed for the random number generator to ensure reproducibility.
 * @var test::N
 * The dimension of the square matrices for this test case.
 * @var test::output_save_file
 * A character array storing the path to the file where the reference output is saved.
 */
struct test {
	int seed;
	int N;
	
	char output_save_file[100];
};

/**
 * @brief Prototype for the user-implemented matrix solver.
 * This function is expected to be defined in one of the solver implementation files.
 */
double* my_solver(int, double *, double *);

/**
 * @brief Prototype for the test execution function.
 * This function runs a specific test case using a given solver.
 */
int run_test(struct test, Solver, float *);

/**
 * @brief Prototype for a function to free allocated matrix data.
 */
void free_data(double **);

/**
 * @brief Prototype for the data generation function.
 * This function allocates and populates matrices with random data based on test parameters.
 */
int generate_data(struct test, double **, int);

/**
 * @brief Prototype for the input file reader.
 * This function reads test case configurations from a file.
 */
int read_input_file(char *, int *, struct test **);
