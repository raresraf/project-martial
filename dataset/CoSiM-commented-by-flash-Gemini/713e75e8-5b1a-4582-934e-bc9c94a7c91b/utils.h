/**
 * @713e75e8-5b1a-4582-934e-bc9c94a7c91b/utils.h
 * @brief Common utilities and abstractions for matrix solver benchmarking.
 * Functional Utility: Provides standardized data types, random number generators, 
 * and test harness definitions for performance evaluation.
 * Domain: HPC Infrastructure.
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

/**
 * Functional Utility: Abstracted function pointer type for matrix solver implementations.
 */
typedef double* (*Solver)(int, double *, double*);

/**
 * Functional Utility: Generates a pseudo-random double-precision value within [-limit, limit].
 */
#define get_rand_double(limit) ((((double)rand()) / RAND_MAX) * (2 * limit) - limit)

/**
 * @brief Configuration parameters for a single solver execution test.
 */
struct test {
	int seed;
	int N;
	
	char output_save_file[100];
};

/**
 * Functional Utility: Forward declarations for solver interface and test infrastructure.
 */
double* my_solver(int, double *, double *);

int run_test(struct test, Solver, float *);

void free_data(double **);

int generate_data(struct test, double **, int);

int read_input_file(char *, int *, struct test **);
