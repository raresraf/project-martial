/**
 * @file utils.h
 * @brief Header definitions for matrix solver testing and benchmarking utilities.
 * Defines function prototypes for generating, executing, and comparing solver configurations.
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

/**
 * @typedef Solver
 * @brief Function pointer type for matrix solver implementations.
 */
typedef double* (*Solver)(int, double *, double*);

/**
 * @def get_rand_double(limit)
 * @brief Generates a random double precision floating point number within a specified symmetric limit.
 */
#define get_rand_double(limit) ((((double)rand()) / RAND_MAX) * (2 * limit) - limit)

/**
 * @struct test
 * @brief Configuration parameters for a solver execution test run.
 */
struct test {
	int seed;
	int N;
	
	char output_save_file[100];
};

double* my_solver(int, double *, double *);

int run_test(struct test, Solver, float *);

void free_data(double **);

int generate_data(struct test, double **, int);

int read_input_file(char *, int *, struct test **);
