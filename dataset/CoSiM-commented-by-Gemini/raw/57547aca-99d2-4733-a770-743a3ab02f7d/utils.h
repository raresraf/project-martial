

/*
 * Module: Utilities
 * @raw/57547aca-99d2-4733-a770-743a3ab02f7d/utils.h
 * Purpose: Shared utils and constants.
 */
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

typedef double* (*Solver)(int, double *, double*);

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
