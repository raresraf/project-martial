/**
 * @6505cf6c-e20c-45bb-bb77-ede140ac5ca6/solver_neopt.c
 * @brief Baseline reference implementation of a matrix expression solver.
 * Functional Utility: Implements Result = (A * B) * B^T + (A^T * A) using unoptimized 
 * iterative loops. Serves as a control for performance benchmarking.
 * Domain: HPC Numerical Baselines.
 */

#include "utils.h"


/**
 * @brief Naive matrix solver kernel.
 * Logic: Step-by-step computation of matrix products using triple-nested loops.
 */
double* my_solver(int N, double *A, double* B) {
	double *AB;
	double *ABB_t;
	double *AtA;
	double *C;
	int i, j, k;

	AB = calloc(N * N, sizeof(double));
	ABB_t = calloc(N * N, sizeof(double));
	AtA = calloc(N * N, sizeof(double));
	C = calloc(N * N, sizeof(double));

	/**
	 * Block Logic: Compute AB = A * B.
	 * Invariant: Exploits A's upper triangular sparsity (k starts at i).
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = i; k < N; k++) {
				AB[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	/**
	 * Block Logic: Compute ABB_t = AB * B^T.
	 * Logic: Transposes B by accessing it as B[j * N + k].
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				ABB_t[i * N + j] += AB[i * N + k] * B[j * N + k];
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
	 * Block Logic: Final summation stage.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i * N + j] = ABB_t[i * N + j] + AtA[i * N + j];
		}
	}

	free(AB);
	free(ABB_t);
	free(AtA);
	return C;
}
