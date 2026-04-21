/**
 * @75135121-c23f-45b4-98b9-962563fdff8b/utils.h
 * @brief Common utilities and abstractions for matrix solver benchmarking.
 *
 * Domain: HPC Infrastructure.
 * Functional Utility: Provides standardized data structures, random number generators, 
 * and test harness function signatures for evaluating matrix expression kernels.
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

/**
 * Functional Utility: Generic function pointer for matrix solver implementations.
 * Enables polymorphic swap-in of naive, optimized, and BLAS-based kernels.
 */
typedef double* (*Solver)(int, double *, double*);

/**
 * Functional Utility: Generates a pseudo-random double in the range [-limit, limit].
 */
#define get_rand_double(limit) ((((double)rand()) / RAND_MAX) * (2 * limit) - limit)

/**
 * @struct test
 * @brief Encapsulates a single benchmarking scenario.
 */
struct test {
	int seed; // Reproducibility factor for random data.
	int N;    // Square matrix dimension.
	
	char output_save_file[100]; // Target path for result serialization.
};

/**
 * Interface Implementation: Forward declarations for core simulation and I/O logic.
 */
double* my_solver(int, double *, double *);

int run_test(struct test, Solver, float *);

void free_data(double **);

int generate_data(struct test, double **, int);

int read_input_file(char *, int *, struct test **);
