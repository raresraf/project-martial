/*
 * Module: @raw/3a541032-bced-45be-bc2e-47ff5af59048/solver_opt.c
 * High-level purpose: Optimized matrix solver implementation.
 */
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	
	register int i = 0;
	register int j = 0;
	register int k = 0;

	register int size = N * N * sizeof(double);

	double *result_AB = malloc(size);
	

	/* Pre-condition: Memory allocated for result_AB. Invariant: Using fast register variables for bounds. */
	for (i = 0; i < N; i++) {
		register int indx = i * N;
		for (j = 0; j < N; j++) {
			register double sum = 0.0;
			for (k = i; k < N; k++) {
				sum += A[indx + k] * B[k * N + j];
			}
			result_AB[indx + j] = sum;
		}
	}

	
	double *C = malloc(size);

	for (i = 0; i < N; i++) {
		register int indx1 = i * N;
		for (j = 0; j < N; j++) {
			register double sum = 0.0;
			register double sumA = 0.0;
			register int jn = j * N;
			for (k = 0; k < N; k++) {
				register int kn = k * N;
				sum += result_AB[indx1 + k] * B[jn + k];
				sumA += A[kn + i] * A[kn + j];
			}
			C[indx1 + j] = sum + sumA;
		}
	}

	printf("OPT SOLVER\n");
	free(result_AB);
	return C;	
}
