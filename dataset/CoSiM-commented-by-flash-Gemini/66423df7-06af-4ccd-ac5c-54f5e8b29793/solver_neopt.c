/**
 * @66423df7-06af-4ccd-ac5c-54f5e8b29793/solver_neopt.c
 * @brief Naive baseline implementation of a matrix expression solver.
 * Functional Utility: Implements Result = (A * B) * B^T + (A^T * A) using basic 
 * loop structures for reference correctness and performance profiling.
 * Domain: HPC Numerical Baselines.
 */

#include "utils.h"


/**
 * @brief Unoptimized matrix solver kernel.
 * Logic: Sequential computation of matrix products using triple-nested loops.
 */
double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	
	double *AB = (double *)calloc(N * N, sizeof(double));
	double *ABBt = (double *)calloc(N * N, sizeof(double));
	double *AtA = (double *)calloc(N * N, sizeof(double));
	double *RES = (double *)calloc(N * N, sizeof(double));
	int i, j, k;

	/**
	 * Block Logic: Compute AB = A * B.
	 * Invariant: k loop respects A's upper triangular sparsity (k starts at i).
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = i; k < N; k++) {
				AB[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	/**
	 * Block Logic: Compute ABBt = AB * B^T.
	 * Logic: Implicitly transposes B by swapping indices (B[j * N + k]).
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				ABBt[i * N + j] += AB[i * N + k] * B[j * N + k];
			}
		}
	}

	/**
	 * Block Logic: Compute AtA = A^T * A.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				AtA[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	/**
	 * Block Logic: Consolidate partial products into the result matrix RES.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			RES[i * N + j] = ABBt[i * N + j] + AtA[i * N + j];
		}
	}

	free(AB);
	free(ABBt);
	free(AtA);
	return RES;
}
