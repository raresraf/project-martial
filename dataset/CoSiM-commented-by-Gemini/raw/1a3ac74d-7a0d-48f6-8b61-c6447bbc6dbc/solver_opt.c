/**
 * @raw/1a3ac74d-7a0d-48f6-8b61-c6447bbc6dbc/solver_opt.c
 * @brief Computes the matrix expression C = A * B * B^T + A^T * A using optimized loops with register variables and local accumulators.
 * * Algorithm: Optimized iterative matrix multiplication. Exploits the upper triangular property of matrix A.
 * * Performance Optimization: Utilizes `register` keyword hints and local scalar accumulators (`sum`) to minimize memory access overhead.
 * Time Complexity: $O(N^3)$ utilizing three nested loops for multiplication.
 * Space Complexity: $O(N^2)$ for dynamically allocated intermediate matrices.
 */

#include <stdlib.h>
#include "utils.h"

/**
 * Functional Utility: Computes the transpose of a square matrix A into B.
 */
void transpose(int N, double *A, double *B) {
	int i, j;
	/**
	 * Block Logic: Iterates over the matrix dimensions to swap rows and columns.
	 */
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
			B[i * N + j] = A[j * N + i];
		}
	}
}

/**
 * Functional Utility: Solves the matrix expression iteratively with optimizations.
 */
double* my_solver(int N, double *A, double *B) {
	printf("OPT SOLVER\n");
	register double *C = (double *) malloc(N * N * sizeof(double));
	
	register double *ABBt = (double *) malloc(N * N * sizeof(double));
	register double *At = (double *) malloc(N * N * sizeof(double));
	register double *AtA = (double *) malloc(N * N * sizeof(double));

	register double *BBt = (double *) malloc(N * N * sizeof(double));
	register double *Bt = (double *) malloc(N * N * sizeof(double));

	transpose(N, B, Bt);
	register int i, j, k;

	/**
	 * Block Logic: Computes the intermediate matrix BBt = B * B^T.
	 * Performance Optimization: Accumulates the product in a local register `sum` before writing to memory to reduce stores.
	 */
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			register double sum = 0.0;
			for (k = 0; k < N; ++k) {
				sum += B[i * N + k] * Bt[k * N + j];
			}
			BBt[i * N + j] = sum;
		}
	}

	/**
	 * Block Logic: Computes ABBt = A * BBt.
	 * Performance Optimization: Uses a local accumulator and exploits the upper triangular property of matrix A by setting inner loop start to `k = i`.
	 */
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			register double sum = 0.0;
			for (k = i; k < N; ++k) {
				sum += A[i * N + k] * BBt[k * N + j];
			}
			ABBt[i * N + j] = sum;
		}
	}

	transpose(N, A, At);

	/**
	 * Block Logic: Computes AtA = A^T * A.
	 * Performance Optimization: Binds the inner loop to compute only non-zero entries based on triangular limits (`k <= i && k <= j`).
	 */
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			register double sum = 0.0;
			for (k = 0; k <= i && k <= j; ++k) {
				sum += At[i * N + k] * A[k * N + j];
			}
			AtA[i * N + j] = sum;
		}
	}

	/**
	 * Block Logic: Sums the intermediate products to form the final matrix C.
	 */
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			C[i * N + j] = ABBt[i * N + j] + AtA[i * N + j]; 
		}
	}
	
	free(ABBt);
	free(At);
	free(AtA);
	free(BBt);
	free(Bt);
	return C;
}
