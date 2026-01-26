/**
 * @file solver_opt.c
 * @brief This file implements a matrix solver (`my_solver`) with manual optimizations
 *        for improved performance compared to a naive implementation, without
 *        relying on external optimized libraries like BLAS.
 *
 * Optimizations typically involve using pointer arithmetic and specific loop
 * access patterns to enhance data locality and reduce memory access overhead.
 * It includes a helper function to compute matrix transposes.
 *
 * Algorithm: Direct implementation of matrix operations with manual optimizations
 *            for cache efficiency and reduced instruction count.
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
 * @brief Implements a matrix solver with manual optimizations for matrix operations.
 *
 * This function calculates a complex matrix expression involving input matrices A and B.
 * It performs various matrix multiplications and additions using explicit nested loops,
 * often incorporating pointer arithmetic and `register` hints to improve memory access
 * patterns and potentially leverage CPU registers for performance.
 *
 * @param N The dimension of the square matrices (N x N).
 * @param A A pointer to the first input square matrix.
 * @param B A pointer to the second input square matrix.
 * @return A pointer to the newly allocated result matrix. The caller is
 *         responsible for freeing this memory.
 *
 * Algorithm: Sequence of matrix operations with manual loop unrolling/optimization strategies.
 * Time Complexity: Dominated by matrix multiplications, O(N^3).
 * Space Complexity: O(N^2) for `first_mul`, `first_mul_aux`, `second_mul`, `At`, and `Bt`.
 */
double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n"); // Indicate which solver is being used.

	// Allocate memory for intermediate result matrices.
	double *first_mul = calloc(N * N, sizeof(double));     // Stores result of A * B (optimized, triangular part).
	double *first_mul_aux = calloc(N * N, sizeof(double)); // Stores (A * B) * Bt + At (final result).
	double *second_mul = calloc(N * N, sizeof(double));    // Stores result of At * A (optimized, triangular part).
	
	// Compute transposes of input matrices B and A.
	double *Bt = get_transpose(B, N); // Transpose of B.
	double *At = get_transpose(A, N); // Transpose of A.

	/**
	 * Block Logic: Computes a partial matrix multiplication `first_mul = A * B`
	 *              with manual optimizations using pointer arithmetic and `register` variables.
	 *              The inner loop bounds (`k = i; k < N`) suggest a multiplication
	 *              involving the upper triangular part of B.
	 * Functional Utility: Calculates an intermediate product for the final expression.
	 */
	for (int i = 0; i < N; ++i) {
		register double *aux = &A[i * N]; // Pointer to the current row of A.
		for (int j = 0; j < N; ++j) {
			register double *collumn = &B[j]; // Pointer to the current column of B.
			register double rez = 0;         // Accumulator for the dot product.

			// Inner loop for dot product, using pointer arithmetic and specific bounds.
			for (int k = i; k < N; ++k, line++, collumn += N) {
				rez += *(line + i) * *(collumn + i * N); // Accessing elements in a specific pattern.
			}
			first_mul[i * N + j] = rez; // Store the result.
		}
	}


	/**
	 * Block Logic: Computes `first_mul_aux = first_mul * Bt`
	 *              with manual optimizations using pointer arithmetic and `register` variables.
	 * Functional Utility: Calculates another intermediate product.
	 */
	for (int i = 0; i < N; ++i) {
		register double *aux = &first_mul[i * N]; // Pointer to the current row of `first_mul`.
		for (int j = 0; j < N; ++j) {
			register double *line = aux;          // Pointer to traverse `first_mul` row.
			register double *collumn = &Bt[j];    // Pointer to traverse `Bt` column.
			register double res = 0;              // Accumulator for the dot product.

			// Inner loop for dot product using pointer arithmetic.
			for (int k = 0; k < N; ++k, line++, collumn += N) {
				res += *line * *collumn; // Perform multiplication and accumulate.
			}
			first_mul_aux[i * N + j] = res; // Store the result.
		}
	}

	/**
	 * Block Logic: Computes a partial matrix multiplication `second_mul = At * A`
	 *              with manual optimizations using pointer arithmetic and `register` variables.
	 *              The inner loop bounds (`k = 0; k <= i`) suggest a multiplication
	 *              involving the lower triangular part of A.
	 * Functional Utility: Calculates an intermediate product for the final expression.
	 */
	for (int i = 0; i < N; ++i) {
		register double *aux = &At[i * N]; // Pointer to the current row of At.
		for (int j = 0; j < N; ++j) {
			register double *line = aux;        // Pointer to traverse At row.
			register double *collumn = &A[j];   // Pointer to traverse A column.
			register double res = 0;            // Accumulator for the dot product.

			// Inner loop for dot product, using pointer arithmetic and specific bounds.
			for (int k = 0; k <= i; ++k, line++, collumn += N) {
				res += *line * *collumn; // Perform multiplication and accumulate.
			}
			second_mul[i * N + j] = res; // Store the result.
		}
	}

	/**
	 * Block Logic: Performs element-wise addition of `second_mul` to `first_mul_aux`.
	 * Functional Utility: Combines the two main intermediate results to form the final result.
	 */
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			first_mul_aux[i * N + j] += second_mul[i * N + j]; // Element-wise addition.
		}
	}

	/**
	 * Block Logic: Frees dynamically allocated memory for intermediate matrices.
	 * Functional Utility: Prevents memory leaks by releasing resources.
	 */
	free(first_mul);
	free(second_mul);
	free(At);
	free(Bt);

	return first_mul_aux;	// Return the final result matrix.
}
