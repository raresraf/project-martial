/**
 * @6d3557c2-b54d-4ed9-8f5a-505bf69b7218/solver_neopt.c
 * @brief Unoptimized baseline implementation of a matrix expression solver.
 * Functional Utility: Implements Result = (A * B) * B^T + (A^T * A) using standard 
 * triple-nested iterative loops.
 * Domain: HPC Numerical Baselines.
 */

#include "utils.h"


/**
 * @brief Naive matrix solver kernel.
 * Logic: Sequential execution of matrix multiplication stages with temporary storage.
 */
double* my_solver(int N, double *A, double* B) {
	
	double *AB = (double *) calloc(N * N, sizeof(double));
	double *ABBt = (double *) calloc(N * N, sizeof(double));
	double *AtA = (double *) calloc(N * N, sizeof(double));
	double *RES = (double *) calloc(N * N, sizeof(double));
	int i, j, k;

	/**
	 * Block Logic: Compute AB = A * B.
	 * Invariant: k starts at i to exploit A's upper triangular sparsity.
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
	 * Logic: Index-swapping on B (B[j * N + k]) to effect transposition.
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
	 * Logic: Accumulates products of elements from columns of A.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				AtA[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	/**
	 * Block Logic: Consolidate partial products into the result matrix.
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
