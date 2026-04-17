
/* @raw/49030feb-a9ca-47ac-b322-44108964dbd0/utils.h: Utilities for matrix solver */
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
