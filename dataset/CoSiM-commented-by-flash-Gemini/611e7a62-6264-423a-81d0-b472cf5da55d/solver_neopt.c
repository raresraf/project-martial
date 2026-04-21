/**
 * @611e7a62-6264-423a-81d0-b472cf5da55d/solver_neopt.c
 * @brief Baseline implementation of a matrix expression solver.
 * Functional Utility: Computes Result = (A * B) * B^T + (A^T * A) using basic 
 * iterative triple-nested loops.
 * Domain: HPC Numerical Baselines.
 */

#include "utils.h"


/**
 * @brief Naive matrix solver kernel.
 * Logic: Implements canonical matrix multiplication algorithm.
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
	 * Block Logic: Compute C = A * B.
	 * Invariant: k loop is adjusted to skip zero elements in upper triangular matrix A.
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
	 * Block Logic: Compute D = C * B^T.
	 * Logic: Accesses B in column-major fashion (B[k + j * N]) to effect transposition.
	 */
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			for(k = 0; k < N; k++) {
				D[i * N + j] +=  C[i * N + k] * B[k + j * N];
			}
		}
	}

	/**
	 * Block Logic: Compute E = D + (A^T * A).
	 * Logic: Aggregates the previous intermediate D with the element-wise Gramian 
	 * contribution of matrix A.
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
