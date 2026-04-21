/**
 * @623b273a-0241-4abf-bd8b-e5af273c535b/solver_neopt.c
 * @brief Baseline implementation of a matrix expression solver.
 * Functional Utility: Implements Result = (A * B) * B^T + (A^T * A) using basic 
 * iterative triple-nested loops without hardware acceleration.
 * Domain: HPC Numerical Baselines.
 */

#include "utils.h"


/**
 * @brief Naive matrix solver kernel.
 * Logic: Stage-based computation of matrix products with explicit buffer management.
 */
double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	double *AAt;
	double *BBt;
	double *RESULT;
	int i, j, k;

	AAt = calloc(N * N, sizeof(*AAt));
	BBt = calloc(N * N, sizeof(*BBt));
	RESULT = calloc(N * N, sizeof(*RESULT));

	/**
	 * Block Logic: Compute BBt = A * B.
	 * Invariant: Exploits the upper triangular property of A (k loop starts at i).
	 */
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			for(k = i; k < N; k++) {
				BBt[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	/**
	 * Block Logic: Compute RESULT = BBt * B^T.
	 * Logic: Accesses matrix B in column-major order to simulate transposition.
	 */
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			for(k = 0; k < N; k++) {
				RESULT[i * N + j] += BBt[i * N + k] * B[j * N + k];
			}
		}
	}

	/**
	 * Block Logic: Compute AAt = A^T * A.
	 * Logic: Computes the dot product of A's columns.
	 */
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			for(k = 0; k < N; k++) {
				AAt[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	/**
	 * Block Logic: Aggregate partial products into the final output matrix.
	 */
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			RESULT[i * N + j] += AAt[i * N + j];
		}
	}

	free(BBt);
	free(AAt);

	return RESULT;
}
