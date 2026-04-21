
#include "utils.h"

/**
 * @file solver_neopt.c
 * @brief Unoptimized (naive) implementation of the matrix expression solver.
 * 
 * Functional Intent: Computes the resulting matrix for the expression:
 * Result = (A^T * A) + A * (B * B^T)
 * where A is an upper triangular matrix.
 * 
 * Algorithm: Triple-nested loop matrix multiplication.
 * Time Complexity: $O(N^3)$.
 * Space Complexity: $O(N^2)$ for intermediate matrices (AtA, BBt, ABBt).
 * 
 * Domain: HPC, Linear Algebra, Reference Implementation.
 */

/**
 * my_solver - Naive implementation using standard C loops.
 */
double* my_solver(int N, double *A, double* B) {
	double *C;
	double *AtA, *BBt, *ABBt;
	int i, j, k;
	
	// Logic: Pre-allocation of workspaces.
	C = malloc(N * N * sizeof(*C));
	AtA = calloc(N * N, sizeof(*AtA));
	BBt = calloc(N * N, sizeof(*BBt));
	ABBt = calloc(N * N, sizeof(*ABBt));
	
	// Guard: Ensure allocation success.
	if (C == NULL || AtA == NULL || BBt == NULL || ABBt == NULL) {
		exit(EXIT_FAILURE);
	}
	
	/**
	 * Block Logic: Compute AtA = A^T * A.
	 * Optimization: Exploits A's upper triangularity (k <= i). 
	 * Handles transposition by accessing column index k*N+i.
	 */
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			for (k = 0; k <= i; ++k) {
				AtA[i * N + j] += A[k *  N + i] * A[k * N + j];
			}
		}
	}

	/**
	 * Block Logic: Compute BBt = B * B^T.
	 * Logic: Computes the symmetric product by accessing B[j][k] as B^T[k][j].
	 */
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			for (k = 0; k < N; ++k) {
				BBt[i * N + j] += B[i *  N + k] * B[j * N + k];
			}
		}
	}

	/**
	 * Block Logic: Compute ABBt = A * BBt.
	 * Optimization: Exploits A's upper triangularity by starting k from i.
	 */
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			for (k = i; k < N; ++k) {
				ABBt[i * N + j] += A[i *  N + k] * BBt[k * N + j];
			}
		}
	}

	/**
	 * Block Logic: Final summation pass.
	 */
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			C[i * N + j] = ABBt[i * N + j] + AtA[i * N + j];
		}
	}

	free(ABBt);
	free(AtA);
	free(BBt);
	return C;
}
