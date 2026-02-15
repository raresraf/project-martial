
/**
 * @file utils.h
 * @brief This header file contains common utility definitions, data structures, and function declarations
 * used across different solver implementations and testing frameworks within the project.
 * It defines types for solver functions, macros for data generation, a structure for test cases,
 * and external function interfaces for managing data, running tests, and I/O operations.
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

/**
 * @typedef Solver
 * @brief Defines a function pointer type for matrix solver functions.
 * A solver function takes the matrix dimension (int N) and two pointers to double-precision
 * matrices (double *A, double *B) as input, and returns a pointer to a newly allocated
 * double-precision result matrix.
 */
typedef double* (*Solver)(int, double *, double*);

/**
 * @brief Macro to generate a random double-precision floating-point number within a specified range.
 * This macro produces a random double value between `-limit` and `+limit`.
 *
 * @param limit The absolute maximum value for the generated random number.
 * @return A double-precision floating-point number in the range [-limit, limit].
 */
#define get_rand_double(limit) ((((double)rand()) / RAND_MAX) * (2 * limit) - limit)

/**
 * @struct test
 * @brief Represents a test case configuration for matrix solver evaluation.
 * Contains parameters necessary to set up and execute a specific test, including
 * a seed for random data generation, the matrix dimension, and a file path for output.
 */
struct test {
	int seed; /**< @brief Seed for random number generation to ensure repeatable test data. */
	int N;    /**< @brief Dimension of the square matrices (N x N) for the test case. */
	
	char output_save_file[100]; /**< @brief File path to save the output of the solver for this test. */
};

double* my_solver(int, double *, double *);

int run_test(struct test, Solver, float *);

void free_data(double **);

int generate_data(struct test, double **, int);

int read_input_file(char *, int *, struct test **);
