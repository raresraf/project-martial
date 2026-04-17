/**
 * @raw/1a3ac74d-7a0d-48f6-8b61-c6447bbc6dbc/solver_neopt.c
 * @brief Computes the matrix expression C = A * B * B^T + A^T * A using naive, unoptimized loops.
 * * Algorithm: Naive iterative matrix multiplication. Exploits the upper triangular property of matrix A.
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
 * Functional Utility: Solves the matrix expression sequentially.
 */
double* my_solver(int N, double *A, double *B) {
	printf("NEOPT SOLVER\n");
	double *C = (double *) calloc(N * N, sizeof(double));
	double *ABBt = (double *) calloc(N * N, sizeof(double));
	double *At = (double *) calloc(N * N, sizeof(double));
	double *AtA = (double *) calloc(N * N, sizeof(double));

	double *BBt = (double *) calloc(N * N, sizeof(double));
	double *Bt = (double *) calloc(N * N, sizeof(double));
	
	transpose(N, B, Bt);
	int i, j, k;

	/**
	 * Block Logic: Computes the intermediate matrix BBt = B * B^T.
	 * Invariant: BBt accumulates the dot product of rows from B.
	 */
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			for (k = 0; k < N; ++k) {
				BBt[i * N + j] += B[i * N + k] * Bt[k * N + j];
			}
		}
	}

	/**
	 * Block Logic: Computes ABBt = A * BBt.
	 * Performance Optimization: The inner loop starts from k = i, exploiting the fact that matrix A is upper triangular (A[i][k] = 0 for k < i).
	 */
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			for (k = i; k < N; ++k) {
				ABBt[i * N + j] += A[i * N + k] * BBt[k * N + j];
			}
		}
	}

	transpose(N, A, At);

	/**
	 * Block Logic: Computes AtA = A^T * A.
	 * Performance Optimization: The inner loop boundary k <= i && k <= j exploits the triangular nature of A and A^T to avoid redundant zero multiplications.
	 */
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			for (k = 0; k <= i && k <= j; ++k) {
				AtA[i * N + j] += At[i * N + k] * A[k * N + j];
			}
		}
	}

	/**
	 * Block Logic: Sums the two intermediate products to form the final matrix C.
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
