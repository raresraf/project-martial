/**
 * @70258bcd-8e0f-49f7-af71-997809f60c20/solver_neopt.c
 * @brief Baseline implementation of a matrix expression solver.
 * Functional Utility: Computes Result = (A * B) * B^T + (A^T * A) using basic 
 * iterative triple-nested loops.
 * Domain: HPC Numerical Baselines.
 */

#include "utils.h"


/**
 * @brief Naive matrix solver kernel.
 * Logic: Sequential computation of matrix products with explicit intermediate storage.
 */
double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	
	double *AB = (double *) calloc(N * N, sizeof(double));
	double *ABBt = (double *) calloc(N * N, sizeof(double));
	double *AtA = (double *) calloc(N * N, sizeof(double));
	double *RES = (double *) calloc(N * N, sizeof(double));
	int i, j, k;

	/**
	 * Block Logic: Compute AB = A * B.
	 * Invariant: k starts at i, exploiting A's upper triangular sparsity.
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
	 * Logic: Accesses B in column-major fashion (B[j * N + k]) to effect transposition.
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
	 * Logic: Computes dot products of columns from matrix A.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				AtA[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	/**
	 * Block Logic: Summation of component matrices into the final result.
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
