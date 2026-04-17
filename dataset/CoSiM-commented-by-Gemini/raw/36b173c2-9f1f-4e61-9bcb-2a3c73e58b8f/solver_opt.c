/**
 * @file solver_opt.c
 * @brief Manually optimized matrix solver implementation.
 * Features register blocking, loop reordering, and cache-friendly data access patterns.
 */

#include "utils.h"


/**
 * @brief Computes C = At * A + A * B * Bt.
 * Allocates memory dynamically and executes matrix operations.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
	int i, j, k;

	double *C = (double*) calloc(N * N, sizeof(double));
	/* @pre Conditional evaluation. @invariant Taken branch maintains control flow invariants. */
	if (C == NULL) {
		printf("Failed calloc!");
		return NULL;
	}

	double *another_C = (double*) calloc(N * N, sizeof(double));
	/* @pre Conditional evaluation. @invariant Taken branch maintains control flow invariants. */
	if (another_C == NULL) {
		printf("Failed calloc!");
		return NULL;
	}

	double *res_A = (double*) calloc(N * N, sizeof(double));
	/* @pre Conditional evaluation. @invariant Taken branch maintains control flow invariants. */
	if (res_A == NULL) {
		printf("Failed calloc!");
		return NULL;
	}

	
	/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
	for (i = 0; i < N; ++i) {
		register double *orig_pa = &A[i * N + i]; /* Non-obvious pointer arithmetic/dereference for optimized memory access */
		/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
		for (j = 0; j < N; ++j) {
			register double *pa = orig_pa; /* Non-obvious pointer arithmetic/dereference for optimized memory access */
    			register double *pb = &B[i * N + j]; /* Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double sum = 0.0;
			/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
			for (k = 0; k < N - i; ++k) {
				sum += *pa * *pb; /* Non-obvious pointer arithmetic/dereference for optimized memory access */
				pa++;
				pb += N; /* Non-obvious pointer arithmetic/dereference for optimized memory access */
			}
			C[i * N + j] = sum;
		}
	}

	
	/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
	for (i = 0; i < N; ++i) {
		register double *orig_pc = &C[i * N + 0]; /* Non-obvious pointer arithmetic/dereference for optimized memory access */
		/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
		for (j = 0; j < N; ++j) {
			register double *pc = orig_pc; /* Non-obvious pointer arithmetic/dereference for optimized memory access */
    		register double *pb = &B[j * N + 0]; /* Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double sum = 0.0;
			/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
			for (k = 0; k < N; ++k) {
				sum += *pc * *pb; /* Non-obvious pointer arithmetic/dereference for optimized memory access */
				pc++;
				pb++;
			}
			another_C[i * N + j] = sum;
		}
	}

	
	/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
	for (k = 0; k < N; ++k) {
		register double *pa = &A[k * N + k]; /* Non-obvious pointer arithmetic/dereference for optimized memory access */
		/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
		for (i = k; i < N; ++i) {
			register double *pa_t = &A[k * N + k]; /* Non-obvious pointer arithmetic/dereference for optimized memory access */
			/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
			for (j = 0; j < N - k; ++j) {
				res_A[i * N + k + j] += *pa_t * *pa; /* Non-obvious pointer arithmetic/dereference for optimized memory access */
				pa_t++;
			}
			pa++;
		}
	}

	
	/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
	for (i = 0; i < N; ++i) {
		register double *orig_pa = &res_A[i * N + 0]; /* Non-obvious pointer arithmetic/dereference for optimized memory access */
		/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
		for (j = 0; j < N; ++j) {
			another_C[i * N + j] += *orig_pa;
			orig_pa++;
		}
	}
	free(res_A);
	free(C);
	return another_C;	
}
