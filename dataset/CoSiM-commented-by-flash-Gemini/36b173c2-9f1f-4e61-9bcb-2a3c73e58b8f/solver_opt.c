/**
 * @file solver_opt.c
 * @brief Implements an optimized matrix solver using manual loop optimizations and pointer arithmetic.
 *
 * This file provides a `my_solver` function that performs a series of matrix operations
 * with explicit optimizations aimed at improving performance over naive implementations.
 * Techniques include careful loop structuring, `register` keyword usage for frequently
 * accessed variables, and direct pointer arithmetic to enhance data locality and reduce
 * memory access overhead. The solver operates on double-precision floating-point matrices.
 *
 * Algorithm: Optimized matrix multiplication and partial matrix construction.
 * Optimization Techniques: Manual loop unrolling, `register` keyword, pointer arithmetic.
 * Time Complexity: Expected to be O(N^3) for N x N matrices, but with a lower constant factor
 *                  compared to naive implementations due to optimizations.
 */

#include "utils.h"
#include <stdlib.h> // For calloc, free
#include <stdio.h>  // For printf (for "OPT SOLVER" print)

/**
 * @brief Solves a matrix problem using optimized matrix operations.
 *
 * This function performs a specific sequence of matrix operations:
 * 1. Calculates a partial product `C` based on matrix `A` and `B` where `A` is implicitly treated
 *    as upper triangular for the multiplication index range.
 * 2. Calculates another partial product `another_C` by multiplying `C` with a modified `B` matrix
 *    (implied transpose-like access).
 * 3. Constructs a `res_A` matrix based on `A` where it appears to be building a lower triangular
 *    product involving `A` and its transpose.
 * 4. Finally, `another_C` is updated by adding elements from `res_A`.
 *
 * The implementation uses `register` keywords and pointer arithmetic for performance optimization.
 *
 * @param N (int): The dimension of the square matrices (N x N).
 * @param A (double*): Pointer to the first input matrix (N x N).
 * @param B (double*): Pointer to the second input matrix (N x N).
 * @return (double*): Pointer to a newly allocated N x N matrix containing the final result, or NULL if memory allocation fails.
 */
double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
	int i, j, k;

	// Block Logic: Allocates memory for temporary matrix C, initialized to zeros.
	double *C = (double*) calloc(N * N, sizeof(double));
	if (C == NULL) {
		printf("Failed calloc!");
		return NULL;
	}

	// Block Logic: Allocates memory for another temporary matrix another_C, initialized to zeros.
	double *another_C = (double*) calloc(N * N, sizeof(double));
	if (another_C == NULL) {
		printf("Failed calloc!");
		free(C); // Inline: Frees previously allocated memory for C to prevent leaks.
		return NULL;
	}

	// Block Logic: Allocates memory for temporary matrix res_A, initialized to zeros.
	double *res_A = (double*) calloc(N * N, sizeof(double));
	if (res_A == NULL) {
		printf("Failed calloc!");
		free(C);
		free(another_C);
		return NULL;
	}

	// Block Logic: First major matrix operation: Calculates C.
	// This appears to be a specialized matrix multiplication C[i][j] = sum(A[i][k] * B[k][j]) where k starts from i.
	// This might correspond to multiplying A by an upper triangular part of B, or A itself being upper triangular.
	for (i = 0; i < N; ++i) {
		// Inline: Pointer to the element A[i][i] (start of the relevant row segment in A).
		register double *orig_pa = &A[i * N + i];
		for (j = 0; j < N; ++j) {
			// Inline: Pointer to the current element in A for the inner product.
			register double *pa = orig_pa;
			// Inline: Pointer to the current element in B for the inner product.
    		register double *pb = &B[i * N + j]; // This seems incorrect based on typical matrix mult. Should be B[k*N+j]
			register double sum = 0.0;
			// Block Logic: Inner loop for calculating the sum for C[i][j].
			// The loop bounds (N - i) and pointer increments suggest a specific pattern,
			// possibly dealing with triangular matrices or specific sub-matrices.
			for (k = 0; k < N - i; ++k) {
				sum += *pa * *pb;
				pa++; // Inline: Moves to the next element in the current row segment of A.
				pb += N; // Inline: Moves to the next row in B (same column).
			}
			C[i * N + j] = sum; // Inline: Stores the calculated sum in C[i][j].
		}
	}

	// Block Logic: Second major matrix operation: Calculates another_C.
	// This appears to be a matrix multiplication another_C[i][j] = sum(C[i][k] * B_transpose[k][j]) or similar.
	for (i = 0; i < N; ++i) {
		// Inline: Pointer to the start of the current row in C.
		register double *orig_pc = &C[i * N + 0];
		for (j = 0; j < N; ++j) {
			// Inline: Pointer to the current element in C for the inner product.
			register double *pc = orig_pc;
			// Inline: Pointer to the start of the current column in B (used as B_transpose row).
    		register double *pb = &B[j * N + 0];
			register double sum = 0.0;
			// Block Logic: Inner loop for calculating the sum for another_C[i][j].
			for (k = 0; k < N; ++k) {
				sum += *pc * *pb;
				pc++; // Inline: Moves to the next element in the current row of C.
				pb++; // Inline: Moves to the next element in the current row of B.
			}
			another_C[i * N + j] = sum; // Inline: Stores the calculated sum in another_C[i][j].
		}
	}

	// Block Logic: Third major matrix operation: Constructs res_A.
	// This loop structure is complex and seems to calculate a product involving A and its transpose,
	// likely related to a lower triangular part or a specific transformation.
	for (k = 0; k < N; ++k) {
		// Inline: Pointer to the diagonal element A[k][k].
		register double *pa = &A[k * N + k];
		for (i = k; i < N; ++i) { // Loop starts from k, suggesting lower triangular or diagonal involvement.
			// Inline: Pointer to A[k][k] for the inner calculation.
			register double *pa_t = &A[k * N + k];
			// Block Logic: Inner loop for calculating elements for res_A.
			// This inner loop iterates N-k times.
			for (j = 0; j < N - k; ++j) {
				// Inline: Accumulates product into res_A[i][k+j].
				res_A[i * N + k + j] += *pa_t * *pa;
				pa_t++; // Inline: Moves to the next element in row k of A.
			}
			pa++; // Inline: Moves to the next element in row i of A.
		}
	}

	// Block Logic: Final matrix operation: Adds res_A to another_C.
	// another_C[i][j] += res_A[i][j].
	for (i = 0; i < N; ++i) {
		// Inline: Pointer to the start of the current row in res_A.
		register double *orig_pa = &res_A[i * N + 0];
		for (j = 0; j < N; ++j) {
			another_C[i * N + j] += *orig_pa; // Inline: Adds the current element from res_A to another_C.
			orig_pa++; // Inline: Moves to the next element in the current row of res_A.
		}
	}
	// Functional Utility: Frees memory allocated for temporary matrices res_A and C.
	free(res_A);
	free(C);
	return another_C; // Functional Utility: Returns the pointer to the dynamically allocated final result matrix another_C.
}
