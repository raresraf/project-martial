/**
 * @file utils.h
 * @brief Utility functions and type definitions for a matrix solver testing framework.
 *
 * This header file declares a set of common utilities, data structures, and function
 * prototypes used for generating test data, running solver implementations, and managing memory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

/**
 * @brief A function pointer type that defines the signature for any matrix solver implementation.
 * @param int The dimension of the matrices.
 * @param double* Pointer to the first input matrix (A).
 * @param double* Pointer to the second input matrix (B).
 * @return A pointer to the resulting matrix.
 */
typedef double* (*Solver)(int, double *, double*);

/**
 * @brief A macro to generate a random double value within a specified symmetric range [-limit, limit].
 */
#define get_rand_double(limit) ((((double)rand()) / RAND_MAX) * (2 * limit) - limit)

/**
 * @brief A structure to hold the parameters for a single test case.
 */
struct test {
	int seed; // The seed for the random number generator.
	int N;    // The dimension of the square matrices.
	
	char output_save_file[100]; // File path to save the output matrix.
};

/**
 * @brief Prototype for the matrix solver function to be implemented by different source files.
 */
double* my_solver(int, double *, double *);

/**
 * @brief Prototype for a function to run a single test case with a given solver.
 */
int run_test(struct test, Solver, float *);

/**
 * @brief Prototype for a function to free allocated matrix data.
 */
void free_data(double **);

/**
 * @brief Prototype for a function to generate random matrix data for a test case.
 */
int generate_data(struct test, double **, int);

/**
 * @brief Prototype for a function to read test configurations from an input file.
 */
int read_input_file(char *, int *, struct test **);