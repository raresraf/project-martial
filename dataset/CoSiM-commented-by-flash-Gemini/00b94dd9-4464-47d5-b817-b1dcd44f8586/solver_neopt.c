/**
 * @file solver_neopt.c
 * @brief This file implements a matrix solver (`my_solver`) using explicit nested loops
 *        for matrix operations. This represents a "non-optimized" or "naive" approach
 *        compared to solutions leveraging highly optimized libraries like BLAS.
 *        It is typically used for comparison, educational purposes, or when optimized
 *        libraries are not available.
 *
 * It includes a helper function to compute matrix transposes.
 *
 * Algorithm: Direct implementation of matrix operations (multiplication, addition)
 *            using triple-nested loops.
 * Time Complexity: Dominated by matrix multiplications, typically O(N^3).
 * Space Complexity: O(N^2) for storing auxiliary matrices and transposes.
 */

#include "utils.h" // Assumed to contain utility functions or definitions.
#include <string.h> // For memcpy().
#include <stdio.h> // For printf().
#include <stdlib.h> // For calloc(), free().


/**
 * @brief Computes the transpose of a square matrix.
 *
 * Allocates memory for the transposed matrix and fills it by swapping
 * row and column indices of the original matrix.
 *
 * @param M A pointer to the original square matrix (stored in row-major order).
 * @param N The dimension of the square matrix (N x N).
 * @return A pointer to the newly allocated and populated transposed matrix.
 *         The caller is responsible for freeing this memory.
 *
 * Algorithm: Direct element-wise transposition.
 * Time Complexity: O(N^2) due to nested loops iterating over all elements.
 * Space Complexity: O(N^2) for the new transposed matrix.
 */
static double *get_transpose(double *M, int N)
{
	// Allocate memory for the transposed matrix, initialized to zeros.
	double *tr = calloc(N * N, sizeof(double));
	/**
	 * Block Logic: Iterates through the original matrix to populate the transposed matrix.
	 * Invariant: `tr[i*N + j]` receives the value from `M[j*N + i]`, effectively
	 *            swapping rows and columns.
	 */
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			tr[i * N + j] = M[j * N + i]; // Perform transposition.
		}
	}
	return tr;
}

/**
 * @brief Implements a matrix solver using direct nested loops for matrix operations.
 *
 * This function calculates a complex matrix expression involving input matrices A and B.
 * It computes intermediate matrix products and sums using explicit triple-nested loops
 * for multiplications and double-nested loops for element-wise addition.
 *
 * @param N The dimension of the square matrices (N x N).
 * @param A A pointer to the first input square matrix.
 * @param B A pointer to the second input square matrix.
 * @return A pointer to the newly allocated result matrix. The caller is
 *         responsible for freeing this memory.
 *
 * Algorithm: Direct implementation of matrix operations.
 * Time Complexity: Dominated by matrix multiplications, O(N^3).
 * Space Complexity: O(N^2) for `second_mul`, `first_mul`, `first_mul_helper`, `At`, and `Bt`.
 */
double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n"); // Indicate which solver is being used.

	// Allocate memory for intermediate matrices.
	double *second_mul = calloc(N * N, sizeof(double));     // Stores result of At * A (triangular part).
	double *At = get_transpose(A, N);                       // Transpose of A.
		
	/**
	 * Block Logic: Computes a partial matrix multiplication `second_mul = At * A`
	 *              considering only the lower triangular part of the inner product for `At`.
	 * Invariant: `second_mul[i*N + j]` accumulates the dot product for a specific element.
	 */
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			for (int k = 0; k <= i; ++k) { // Loop only up to `i`, implying a triangular matrix operation.
				second_mul[i * N + j] += At[i * N + k] * A[k * N + j];
			}
		}
	}

	// Allocate memory for further intermediate matrices.
	double *first_mul = calloc(N * N, sizeof(double));      // Stores result of A * B (triangular part).
	double *first_mul_helper = calloc(N * N, sizeof(double)); // Stores result of first_mul * Bt.
	double *Bt = get_transpose(B, N);                       // Transpose of B.
	
	/**
	 * Block Logic: Computes a partial matrix multiplication `first_mul = A * B`
	 *              considering only the upper triangular part of the inner product for `B`.
	 * Invariant: `first_mul[i*N + j]` accumulates the dot product for a specific element.
	 */
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			for (int k = i; k < N; ++k) { // Loop from `i` to N, implying an upper triangular matrix operation.
				first_mul[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	/**
	 * Block Logic: Computes the matrix multiplication `first_mul_helper = first_mul * Bt`.
	 * Invariant: `first_mul_helper[i*N + j]` stores the dot product of row `i` from `first_mul`
	 *            and column `j` from `Bt`.
	 */
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			for (int k = 0; k < N; ++k) {
				first_mul_helper[i * N + j] += first_mul[i * N + k] * Bt[k * N + j];
			}
		}
	}

	// Copy the contents of `first_mul_helper` to `first_mul` to prepare for final addition.
	memcpy(first_mul, first_mul_helper, N * N * sizeof(double));
	
	/**
	 * Block Logic: Performs element-wise addition of `second_mul` to `first_mul`.
	 * Invariant: Each element of `first_mul` is updated with the sum of itself
	 *            and the corresponding element from `second_mul`.
	 */
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			first_mul[i * N + j] += second_mul[i * N + j]; // Element-wise addition.
		}
	}

	/**
	 * Block Logic: Frees dynamically allocated memory for intermediate matrices.
	 * Functional Utility: Prevents memory leaks by releasing resources.
	 */
	free(first_mul_helper);
	free(At);
	free(Bt);
	free(second_mul);

	return first_mul; // Return the final result matrix.
}
