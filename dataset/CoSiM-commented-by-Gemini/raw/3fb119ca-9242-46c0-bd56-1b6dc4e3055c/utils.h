/**
 * @file utils.h
 * @brief Common utilities and definitions for matrix solver testing.
 *
 * Contains function prototypes and structures for testing the performance and correctness
 * of matrix multiplication solvers.
 */

// Module Level: Utils Header
// @raw/3fb119ca-9242-46c0-bd56-1b6dc4e3055c/utils.h

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

/**
 * @brief Function pointer type for matrix solvers.
 */
typedef double* (*Solver)(int, double *, double*);

/**
 * @brief Macro to generate a random double within a specified limit.
 */
#define get_rand_double(limit) ((((double)rand()) / RAND_MAX) * (2 * limit) - limit)

/**
 * @brief Configuration parameters for a solver test run.
 */
struct test {
	int seed;
	int N;
	
	char output_save_file[100];
};

/**
 * @brief Core solver function to compute matrix operations.
 *
 * @param N Matrix dimension.
 * @param A First input matrix.
 * @param B Second input matrix.
 * @return Pointer to the resulting matrix.
 */
double* my_solver(int, double *, double *);

/**
 * @brief Executes a test run for a given solver.
 */
int run_test(struct test, Solver, float *);

/**
 * @brief Frees dynamically allocated double pointer array.
 */
void free_data(double **);

/**
 * @brief Generates random data for the input matrices.
 */
int generate_data(struct test, double **, int);

/**
 * @brief Reads test configuration from an input file.
 */
int read_input_file(char *, int *, struct test **);
