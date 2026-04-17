/**
 * @raw/1f1da956-27fc-4fad-8b7c-625d0cebc41b/utils.h
 * @brief Common definitions and declarations for the matrix solver evaluation suite.
 */

#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

/**
 * Functional Utility: Typedef for a solver function pointer signature.
 */
typedef double* (*Solver)(int, double *, double*);

/**
 * Functional Utility: Macro to generate a random double precision float within a symmetric limit.
 */
#define get_rand_double(limit) ((((double)rand()) / RAND_MAX) * (2 * limit) - limit)

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

#endif /* UTILS_H */