

/**
 * @file utils.h
 * @brief Utility functions and data structures for a matrix solver testing framework.
 *
 * This header file defines the common interface for matrix solvers, data structures
 * for test cases, and prototypes for functions related to test execution, data
 * generation, and file I/O. It serves as a central point for the components
 * of the testing harness.
 */
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

/**
 * @brief A function pointer type representing a generic matrix solver.
 *
 * A Solver takes the matrix dimension N and two input matrices (A and B) and
 * returns a pointer to the resulting solution matrix.
 */
typedef double* (*Solver)(int, double *, double*);

/**
 * @def get_rand_double(limit)
 * @brief Generates a random double within a specified symmetric range.
 * @param limit The boundary of the range [-limit, limit].
 * @return A random double-precision floating-point number.
 */
#define get_rand_double(limit) ((((double)rand()) / RAND_MAX) * (2 * limit) - limit)

/**
 * @struct test
 * @brief Defines a test case for the matrix solver.
 *
 * This structure holds all the parameters needed to define and run a single
 * test, including the seed for randomization, the size of the matrices, and
 * the file path for saving the output.
 */
struct test {
	int seed; ///< The seed for the random number generator.
	int N; ///< The dimension of the square matrices.
	
	char output_save_file[100]; ///< Path to the file where the result matrix should be saved.
};

/**
 * @brief The solver function to be implemented by different versions (BLAS, opt, neopt).
 */
double* my_solver(int, double *, double *);

/**
 * @brief Runs a specific test case using a given solver.
 */
int run_test(struct test, Solver, float *);

/**
 * @brief Frees the memory allocated for matrix data.
 */
void free_data(double **);

/**
 * @brief Generates random input matrices for a test case.
 */
int generate_data(struct test, double **, int);

/**
 * @brief Reads test configurations from an input file.
 */
int read_input_file(char *, int *, struct test **);
