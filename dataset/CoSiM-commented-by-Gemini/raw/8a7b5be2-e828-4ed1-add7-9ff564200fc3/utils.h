/**
 * @raw/8a7b5be2-e828-4ed1-add7-9ff564200fc3/utils.h
 * @brief Common utility types, macros, and function prototypes for the dense matrix operations suite.
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

/**
 * Functional Utility: Defines a standardized function pointer for interchangeable solver backends.
 */
typedef double* (*Solver)(int, double *, double*);

/**
 * Inline: Generates a uniformly distributed random floating-point value within [-limit, limit] 
 * for dataset initialization and fuzzing.
 */
#define get_rand_double(limit) ((((double)rand()) / RAND_MAX) * (2 * limit) - limit)

/**
 * Functional Utility: Wraps parameters required to orchestrate a distinct matrix testing run.
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
