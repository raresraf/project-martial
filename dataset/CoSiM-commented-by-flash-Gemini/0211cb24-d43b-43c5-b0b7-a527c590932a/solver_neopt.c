/**
 * @file solver_neopt.c
 * @brief Implements a "non-optimized" matrix solver using basic C loops for matrix operations.
 * This file provides a baseline implementation for a matrix computation task,
 * intentionally avoiding specialized optimization libraries like BLAS or manual performance tuning.
 * It serves as a reference for understanding the computational steps and for performance
 * comparison against highly optimized versions.
 * Algorithm: The solver performs a sequence of matrix operations:
 *   1. Transposes input matrices `A` and `B` into `at` and `bt` respectively.
 *   2. Computes `res1 = A * B`, treating `A` as an upper triangular matrix implicitly by loop bounds.
 *   3. Computes `res2 = res1 * B_T` using the transposed `B_T` (`bt`).
 *   4. Computes `res3 = A_T * A`, using the transposed `A_T` (`at`), and treating `A` as lower triangular implicitly by loop bounds.
 *   5. Computes the final result `res = res2 + res3`.
 * Optimization: This implementation relies solely on the compiler's default optimizations for nested loops.
 * It does not employ specific techniques like loop unrolling, cache blocking, or SIMD instructions.
 * Time Complexity: Dominated by the explicit three-nested-loop matrix multiplications, resulting in an $O(N^3)$
 * complexity for operations on $N \times N$ matrices.
 * Space Complexity: $O(N^2)$ for storing the input and several intermediate result matrices (`at`, `bt`, `res1`, `res2`, `res3`, `res`).
 */

#include "utils.h"
#include <stdlib.h>

double* my_solver(int N, double *A, double *B) {
	int i, j, k;

	printf("NEOPT SOLVER\n");

	// Block Logic: Allocate memory for the transpose of matrix A ('at').
	// Precondition: 'N * N' elements are allocated.
	// Invariant: 'at' is initialized with zeros upon successful allocation.
	double *at = calloc(N * N, sizeof(double));
	// Conditional Logic: Handle memory allocation failure.
	if (at == NULL)
		exit(EXIT_FAILURE);

	// Block Logic: Allocate memory for the transpose of matrix B ('bt').
	// Precondition: 'N * N' elements are allocated.
	// Invariant: 'bt' is initialized with zeros upon successful allocation.
	double *bt = calloc(N * N, sizeof(double));
	// Conditional Logic: Handle memory allocation failure.
	if (bt == NULL)
		exit(EXIT_FAILURE);

	// Block Logic: Allocate memory for the first intermediate result matrix ('res1').
	double *res1 = calloc(N * N, sizeof(double));
	// Conditional Logic: Handle memory allocation failure.
	if (res1 == NULL)
		exit(EXIT_FAILURE);

	// Block Logic: Allocate memory for the second intermediate result matrix ('res2').
	double *res2 = calloc(N * N, sizeof(double));
	// Conditional Logic: Handle memory allocation failure.
	if (res2 == NULL)
		exit(EXIT_FAILURE);

	// Block Logic: Allocate memory for the third intermediate result matrix ('res3').
	double *res3 = calloc(N * N, sizeof(double));
        // Conditional Logic: Handle memory allocation failure.
        if (res3 == NULL)
                exit(EXIT_FAILURE);

	// Block Logic: Allocate memory for the final result matrix ('res').
	double *res = calloc(N * N, sizeof(double));
        // Conditional Logic: Handle memory allocation failure.
        if (res == NULL)
                exit(EXIT_FAILURE);


	// Block Logic: Compute transposes of matrices A and B.
	// This loop iterates through A and B, storing their transposes in 'at' and 'bt' respectively.
	// Invariant: After this block, at[j*N + i] will hold A[i*N + j] (A^T) and bt[j*N + i] will hold B[i*N + j] (B^T).
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++) {
			// Inline: Store A[i][j] into at[j][i] (column-major access for transpose).
			at[j * N + i] = A[i *  N + j];
			// Inline: Store B[i][j] into bt[j][i] (column-major access for transpose).
			bt[j * N + i] = B[i *  N + j];
		}

	// Block Logic: Compute res1 = A * B (upper triangular part of A implicitly).
	// This block performs matrix multiplication, accumulating products into 'res1'.
	// Precondition: 'A' and 'B' are N x N matrices.
	// Invariant: 'res1[i * N + j]' holds the sum of products for element (i, j).
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			// Invariant: k starts from 'i', implying an upper triangular access pattern for matrix A.
			for (k = i; k < N; k++) {
				res1[i * N + j] += A[i * N + k]
					* B[k * N + j];
			}

	// Block Logic: Compute res2 = res1 * B_T.
	// This block performs matrix multiplication between 'res1' and the transpose of 'B' ('bt').
	// Precondition: 'res1' and 'bt' are N x N matrices.
	// Invariant: 'res2[i * N + j]' holds the sum of products for element (i, j).
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			for (k = 0; k < N; k++) {
				res2[i * N + j] += res1[i * N + k]
					* bt[k * N + j];
			}

	// Block Logic: Compute res3 = A_T * A (lower triangular part of A implicitly).
	// This block performs matrix multiplication between the transpose of 'A' ('at') and 'A'.
	// Precondition: 'at' and 'A' are N x N matrices.
	// Invariant: 'res3[i * N + j]' holds the sum of products for element (i, j).
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			// Invariant: k iterates up to 'j', implying a lower triangular access pattern for the second matrix.
			for (k = 0; k <= j; k++) {
				res3[i * N + j] += at[i * N + k]
					* A[k * N + j];
			}

	// Block Logic: Compute final result: res = res2 + res3.
	// This loop performs element-wise addition of the two intermediate result matrices.
	// Precondition: 'res2' and 'res3' are N x N matrices.
	// Invariant: 'res[i * N + j]' holds the sum of 'res2[i * N + j]' and 'res3[i * N + j]'.
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++) {
			res[i * N + j] = res2[i * N + j] + res3[i * N + j];
		}

	// Block Logic: Free memory allocated for intermediate matrices.
	free(at);
	free(bt);
	free(res1);
	free(res2);
	free(res3);
	// Functional Utility: Return the final computed result matrix.
	return res;
}
