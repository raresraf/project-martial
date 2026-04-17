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
	
	register int i = 0;
	register int j = 0;
	register int k = 0;

	double *fst = malloc(N * N * sizeof(double));
	
	/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
	for(i = 0; i < N; i++){
  		double *orig_pa = &A[i * N]; /* Non-obvious pointer arithmetic/dereference for optimized memory access */
  		/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
  		for(j = 0; j < N; j++){
    		double *pa = orig_pa; /* Non-obvious pointer arithmetic/dereference for optimized memory access */
    		double *pb = &B[j]; /* Non-obvious pointer arithmetic/dereference for optimized memory access */
    		register double suma = 0;
    		/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
    		for(k = 0; k < N; k++){
      			suma += *pa * *pb; /* Non-obvious pointer arithmetic/dereference for optimized memory access */
      			pa++;
      			pb += N; /* Non-obvious pointer arithmetic/dereference for optimized memory access */
    		}
    		fst[i * N + j] = suma;
  		}
	}

	double *snd = malloc(N * N * sizeof(double));

	/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
	for (i = 0; i < N; i++) {
		register int index = i * N;
		/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
		for (j = 0; j < N; j++) {
			register double sum = 0;
			/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
			for (k = 0; k < N; k++) {
				sum += fst[index + k] * B[j * N + k];
			}
			snd[index + j] = sum;
		}
	}

	double *third = malloc(N * N * sizeof(double));
	
	/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
	for (i = 0; i < N; i++) {
		register int index = i * N;
		/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
		for (k = 0; k < N; k++) {
			register int kn = k * N;
			/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
			for (j = 0; j < N; j++) {
				third[index + j] += A[kn + i] * A[kn + j];
			}
		}
	}

	double *C = malloc(N * N * sizeof(double));

	/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
	for (i = 0; i < N; i++) {
		register int index = i * N;
		/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
		for (j = 0 ; j < N; j++) {
			register int index_j = index + j;
			C[index_j] += third[index_j] + snd[index_j];
		}
	}

	free(fst);
	free(snd);
	free(third);

	return C;	
}

