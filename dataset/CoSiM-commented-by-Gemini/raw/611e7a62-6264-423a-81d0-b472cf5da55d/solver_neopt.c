/**
 * @file solver_neopt.c
 * @brief Straightforward array index mapped solver sequence.
 */
#include "utils.h"

/**
 * @brief Simple multiplication routines via loop nesting.
 */
double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	double *C;
	double *D;
	double *E;
	int i, j, k;
	
	C = calloc(N*N, sizeof(double));
	D = calloc(N*N, sizeof(double));
	E = calloc(N*N, sizeof(double));

	/**
	 * @pre Arrays prepared.
	 * @post A and B mapped to C.
	 */
	for(i = 0; i < N; i++){
		for (j = 0; j < N; j++) {
			int var = 0;
			if (i >= j) {
				var = i;
			}
			for (k = var; k < N; k++) {
				C[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	/**
	 * @pre C loaded.
	 * @post D aggregates from C combinations.
	 */
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			for(k = 0; k < N; k++) {
				D[i * N + j] +=  C[i * N + k] * B[k + j * N];
			}
		}
	}

	/**
	 * @pre Secondary iteration completed.
	 * @post Output E sums transpose items and previous D.
	 */
	for(i = 0; i < N; i++){
		for (j = 0; j < N; j++) {
			for (k = 0; k <= j; k++) {
				E[i * N + j] += A[k * N + i] * A[k * N + j];
			}
			E[i * N + j] += D[i * N + j];
		}
	}

	free(C);
	free(D);

 	return E;
}
