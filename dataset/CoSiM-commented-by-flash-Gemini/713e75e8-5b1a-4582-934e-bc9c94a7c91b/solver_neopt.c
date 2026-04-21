/**
 * @713e75e8-5b1a-4582-934e-bc9c94a7c91b/solver_neopt.c
 * @brief Naive implementation of a matrix expression solver.
 * Functional Utility: Implements Result = (A * B) * B^T + (A^T * A) using basic 
 * iterative triple-nested loops.
 * Domain: HPC Numerical Baselines.
 */

#include "utils.h"


/**
 * @brief Unoptimized matrix solver kernel.
 * Logic: Computes partial matrix products sequentially with explicit buffer allocation.
 */
double* my_solver(int N, double *A, double* B) {
	
	double *AB = calloc(sizeof(double), N * N);
	double *ABBt = calloc(sizeof(double), N * N);
	double *AtA = calloc(sizeof(double), N * N);
	double *RES = calloc(sizeof(double), N * N);

	int i, j, k;

	/**
	 * Block Logic: Compute AB = A * B.
	 * Invariant: k loop starts at i to leverage A's upper triangular sparsity.
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
	 * Logic: Transposes matrix B by accessing its elements as B[j * N + k].
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
	 * Logic: Multiplies transposed A by A using standard dot-product accumulation.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				AtA[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	/**
	 * Block Logic: Consolidate term results into final output.
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
