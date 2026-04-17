/**
 * @file solver_neopt.c
 * @brief A non-optimized, baseline implementation of a matrix solver.
 * @details This file provides a straightforward, unoptimized solution for the matrix equation
 * C = A * B * B' + A' * A, where A is an upper triangular matrix. It serves as a reference
 * for correctness and a baseline for performance comparison against optimized versions.
 */

#include <stdlib.h>
#include "utils.h"

/**
 * @brief Computes the transpose of a square matrix.
 * @param N The dimension of the square matrix.
 * @param A The input matrix (N x N) to be transposed.
 * @param B The output matrix (N x N) where the transpose will be stored.
 *
 * This function performs an out-of-place matrix transposition.
 * Time Complexity: O(N^2)
 */
void transpose(int N, double *A, double *B) {
	int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
			B[i * N + j] = A[j * N + i];
		}
	}
}

/**
 * @brief Solves the matrix equation C = A * B * B' + A' * A using a non-optimized approach.
 * @param N The dimension of the square matrices.
 * @param A A pointer to the input upper triangular matrix A (N x N).
 * @param B A pointer to the input matrix B (N x N).
 * @return A pointer to the resulting matrix C (N x N).
 *
 * @details The function follows these steps:
 * 1. Transposes B to get B'.
 * 2. Calculates BBt = B * B'.
 * 3. Calculates ABBt = A * BBt, exploiting the upper triangular nature of A.
 * 4. Transposes A to get A'.
 * 5. Calculates AtA = A' * A, exploiting the triangular nature of A and A'.
 * 6. Computes the final result C = ABBt + AtA.
 *
 * This implementation uses nested loops for all matrix operations and allocates
 * intermediate matrices, making it a clear but inefficient baseline.
 */
double* my_solver(int N, double *A, double *B) {
	printf("NEOPT SOLVER
");
	double *C = (double *) calloc(N * N, sizeof(double));
	double *ABBt = (double *) calloc(N * N, sizeof(double));
	double *At = (double *) calloc(N * N, sizeof(double));
	double *AtA = (double *) calloc(N * N, sizeof(double));

	double *BBt = (double *) calloc(N * N, sizeof(double));
	double *Bt = (double *) calloc(N * N, sizeof(double));
	
	// Step 1: Transpose B to get B'.
	transpose(N, B, Bt);
	int i, j, k;

	
	/**
	 * Block Logic: Step 2: Compute BBt = B * B'.
	 * This is a standard matrix-matrix multiplication.
	 * Time Complexity: O(N^3)
	 */
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			for (k = 0; k < N; ++k) {
				BBt[i * N + j] += B[i * N + k] * Bt[k * N + j];
			}
		}
	}

	
	/**
	 * Block Logic: Step 3: Compute ABBt = A * BBt.
	 * Since A is an upper triangular matrix, the inner loop for k starts from i.
	 * This optimization reduces the number of multiplications.
	 * Time Complexity: O(N^3), but with a lower constant factor than full matrix multiplication.
	 */
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			for (k = i; k < N; ++k) {
				ABBt[i * N + j] += A[i * N + k] * BBt[k * N + j];
			}
		}
	}

	// Step 4: Transpose A to get A'.
	transpose(N, A, At);

	
	/**
	 * Block Logic: Step 5: Compute AtA = A' * A.
	 * A' is lower triangular and A is upper triangular. The loop for k reflects
	 * this structure to avoid unnecessary computations with zero elements.
	 * The condition k <= i corresponds to the lower triangular At.
	 * The condition k <= j corresponds to the upper triangular A.
	 * Time Complexity: O(N^3), with a lower constant factor.
	 */
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			for (k = 0; k <= i && k <= j; ++k) {
				AtA[i * N + j] += At[i * N + k] * A[k * N + j];
			}
		}
	}

	/**
	 * Block Logic: Step 6: Compute C = ABBt + AtA.
	 * This is a simple element-wise matrix addition.
	 * Time Complexity: O(N^2)
	 */
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			C[i * N + j] = ABBt[i * N + j] + AtA[i * N + j]; 
		}
	}
	
	// Free all dynamically allocated intermediate matrices.
	free(ABBt);
	free(At);
	free(AtA);
	free(BBt);
	free(Bt);

	return C;
}
