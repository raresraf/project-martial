/**
 * @file solver_opt.c
 * @brief Implements a manually "optimized" matrix solver using pure C.
 * This file provides an optimized implementation for a matrix computation task,
 * leveraging manual C-level optimizations to improve performance over a naive approach.
 * Techniques include `register` keyword usage for frequently accessed variables and
 * direct pointer arithmetic for efficient array traversal.
 * Algorithm: The solver performs a sequence of matrix operations similar to the non-optimized version:
 *   1. Computes `A_T` (transpose of A) and `B_T` (transpose of B) using direct pointer manipulation.
 *   2. Computes `res1 = A * B`, implicitly treating `A` as an upper triangular matrix and using pointer arithmetic for access.
 *   3. Computes `res2 = res1 * B_T` using pointer arithmetic for access.
 *   4. Computes `res3 = A_T * A`, implicitly treating `A` as lower triangular and using pointer arithmetic for access.
 *   5. Computes the final result `res = res2 + res3`.
 * Optimization:
 *   - `register` keyword: Hints to the compiler to store loop counters and frequently used pointers in CPU registers for faster access.
 *   - Pointer arithmetic: Direct manipulation of pointers to traverse arrays, potentially reducing address calculation overhead.
 *   - Loop structure: Designed to potentially enhance cache locality by accessing contiguous memory blocks.
 * Time Complexity: Theoretically $O(N^3)$ for $N \times N$ matrices due to three-nested-loop matrix multiplications.
 * However, the manual optimizations aim to significantly reduce the constant factor, leading to improved practical performance.
 * Space Complexity: $O(N^2)$ for storing the input and several intermediate result matrices (`at`, `bt`, `res1`, `res2`, `res3`, `res`).
 */

#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	// Inline: Declare loop counters as 'register' to suggest CPU register storage for performance optimization.
	register int i, j, k;

	printf("OPT SOLVER\n");

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


	// Block Logic: Compute transposes of matrices A and B using pointer arithmetic.
	// This block iterates through A and B, storing their transposes in 'at' and 'bt' respectively.
	// Optimization: Uses explicit pointers and increments to access matrix elements, aiming for faster memory access.
	// Invariant: After this block, at[j*N + i] will hold A[i*N + j] (A^T) and bt[j*N + i] will hold B[i*N + j] (B^T).
	for (i = 0; i < N; i++) {

		// Inline: Pointers for row-wise access in original matrices A and B.
		register double *p_at = at + i; // Inline: Pointer for column-wise writing to 'at' (transpose of A).
		register double *p_bt = bt + i; // Inline: Pointer for column-wise writing to 'bt' (transpose of B).
		register double *ptA = A + i * N; // Inline: Pointer for row 'i' of matrix A.
		register double *ptB = B + i * N; // Inline: Pointer for row 'i' of matrix B.

		for (j = 0; j < N; j++) {
			*p_at = *ptA; // Inline: Copy A[i][j] to at[j][i].
			*p_bt = *ptB; // Inline: Copy B[i][j] to bt[j][i].
			p_at += N; // Inline: Move to the next row in the transpose matrix (effectively next column in original).
			p_bt += N; // Inline: Move to the next row in the transpose matrix (effectively next column in original).
			ptA++; // Inline: Move to the next element in the current row of A.
			ptB++; // Inline: Move to the next element in the current row of B.
		}
	}

	// Block Logic: Compute res1 = A * B (upper triangular part of A implicitly).
	// This block performs matrix multiplication, accumulating products into 'res1'.
	// Optimization: Uses explicit pointer arithmetic for matrix element access.
	// Precondition: 'A' and 'B' are N x N matrices.
	// Invariant: 'res1[i * N + j]' holds the sum of products for element (i, j).
	for (i = 0; i < N; i++) {
		// Inline: Pointer to the diagonal element of current row 'i' in matrix A.
		register double *pt_to_A = &A[i * N + i];
		// Inline: Pointer to the start of current row 'i' in result matrix 'res1'.
		register double *p_res1 = &res1[i * N];
		for (j = 0; j < N; j++) {
			// Inline: Initialize sum for the current element res1[i][j].
			register double sum = 0.0;
			// Inline: Pointer for current element in matrix A (row 'i', starting from 'i').
			register double *pa = pt_to_A;
			// Inline: Pointer for current element in matrix B (row 'i', column 'j').
			register double *pb = &B[i*N + j];
			// Invariant: k starts from 'i', implying an upper triangular access pattern for matrix A.
			for (k = i; k < N; k++) {
				sum += *pa * *pb; // Inline: Accumulate product of elements.
				pa++; // Inline: Move to next element in current row of A.
				pb += N; // Inline: Move to next row in current column of B.
			}
			*(p_res1 + j) = sum; // Inline: Store computed sum into res1[i][j].
		}
	}

	// Block Logic: Compute res2 = res1 * B_T.
	// This block performs matrix multiplication between 'res1' and the transpose of 'B' ('bt').
	// Optimization: Uses explicit pointer arithmetic for matrix element access.
	// Precondition: 'res1' and 'bt' are N x N matrices.
	// Invariant: 'res2[i * N + j]' holds the sum of products for element (i, j).
	for (i = 0; i < N; i++) {
		// Inline: Pointer to the start of current row 'i' in matrix 'res1'.
		register double *pt_to_res1 = &res1[i * N];
		for (j = 0; j < N; j++) {
			// Inline: Initialize sum for the current element res2[i][j].
			register double sum = 0.0;
			// Inline: Pointer for current element in matrix 'res1' (row 'i').
			register double *pres1 = pt_to_res1;
			// Inline: Pointer for current element in transposed matrix 'bt' (column 'j').
			register double *pbt = &bt[j];
			for (k = 0; k < N; k++) {
				sum += *pres1 * *pbt; // Inline: Accumulate product of elements.
				pres1++; // Inline: Move to next element in current row of res1.
				pbt += N; // Inline: Move to next row in current column of bt.
			}
			res2[i * N + j]= sum; // Inline: Store computed sum into res2[i][j].
		}
	}

	// Block Logic: Compute res3 = A_T * A (lower triangular part of A implicitly).
	// This block performs matrix multiplication between the transpose of 'A' ('at') and 'A'.
	// Optimization: Uses explicit pointer arithmetic for matrix element access.
	// Precondition: 'at' and 'A' are N x N matrices.
	// Invariant: 'res3[i * N + j]' holds the sum of products for element (i, j).
	for (i = 0; i < N; i++) {
		// Inline: Pointer to the start of current row 'i' in transposed matrix 'at'.
		register double *pt_to_at = &at[i * N ];
		for (j = 0; j < N; j++) {
			// Inline: Initialize sum for the current element res3[i][j].
			register double sum = 0.0;
			// Inline: Pointer for current element in transposed matrix 'at' (row 'i').
			register double *pat = pt_to_at;
			// Inline: Pointer for current element in matrix A (column 'j').
			register double *pA = &A[j];
			// Invariant: k iterates up to 'j', implying a lower triangular access pattern for the second matrix.
			for (k = 0; k <= j; k++) {
				sum += *pat * *pA; // Inline: Accumulate product of elements.
				pat++; // Inline: Move to next element in current row of at.
				pA += N; // Inline: Move to next row in current column of A.
			}
			res3[i * N + j] = sum ; // Inline: Store computed sum into res3[i][j].
		}
	}

	// Block Logic: Compute final result: res = res2 + res3.
	// This loop performs element-wise addition of the two intermediate result matrices.
	// Precondition: 'res2' and 'res3' are N x N matrices.
	// Invariant: 'res[i * N + j]' holds the sum of 'res2[i * N + j]' and 'res3[i * N + j]'.
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			res[i * N + j] = res2[i * N + j] + res3[i * N + j];
		}
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
