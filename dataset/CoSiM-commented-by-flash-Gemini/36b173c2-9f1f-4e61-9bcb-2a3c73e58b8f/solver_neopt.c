/**
 * @file solver_neopt.c
 * @brief Implements a non-optimized matrix solver using naive matrix operations.
 *
 * This file provides a `my_solver` function that performs matrix operations,
 * including transposition and multiplication, using basic nested loops.
 * This implementation serves as a baseline, likely for performance comparison
 * against optimized versions (e.g., using BLAS).
 * The solver operates on double-precision floating-point matrices.
 *
 * Algorithm: Naive matrix multiplication (triple-nested loops) and matrix transposition.
 * Time Complexity: O(N^3) for matrix multiplications, O(N^2) for transposition,
 *                  resulting in an overall O(N^3) complexity for N x N matrices.
 */

#include "utils.h"
#include <stdlib.h> // For calloc, free
#include <stdio.h>  // For printf

/**
 * @brief Macro to find the minimum of two values.
 * @param x (int): The first value.
 * @param y (int): The second value.
 * @return (int): The minimum of x and y.
 */
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

/**
 * @brief Computes the transpose of a square matrix.
 *
 * Allocates new memory for the transposed matrix and fills it by swapping
 * row and column indices of the original matrix.
 *
 * @param N (int): The dimension of the square matrix (N x N).
 * @param to_be_transposed (double*): Pointer to the original matrix.
 * @return (double*): Pointer to the newly allocated transposed matrix, or NULL if memory allocation fails.
 */
double* my_transpose(int N, double* to_be_transposed) {
	int i, j;
	// Block Logic: Allocates memory for the transposed matrix, initialized to zeros.
	double *transpose = (double*) calloc(N * N, sizeof(double));
	// Block Logic: Checks if memory allocation was successful.
	if (transpose == NULL) {
		printf("Failed calloc for transpose matrix!\n");
		return NULL;
	}

	// Block Logic: Iterates through the original matrix and fills the transposed matrix.
	// Invariant: transpose[i][j] = to_be_transposed[j][i].
	for (i  = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			transpose[i * N + j] = to_be_transposed[j * N + i];
		}
	}
	return transpose;
}

/**
 * @brief Solves a matrix problem using naive, non-optimized matrix operations.
 *
 * This function performs a series of matrix operations, including transposition
 * and multiplication, using direct nested loops. It is explicitly "non-optimized"
 * for demonstration or comparison purposes. The specific computation is
 * (A * B^T) * B^T + (A^T * A) where A^T * A is a lower triangular multiply.
 *
 * @param N (int): The dimension of the square matrices (N x N).
 * @param A (double*): Pointer to the first input matrix (N x N).
 * @param B (double*): Pointer to the second input matrix (N x N).
 * @return (double*): Pointer to a newly allocated N x N matrix containing the result, or NULL if memory allocation fails.
 */
double* my_solver(int N, double *A, double* B) {
	int i, j, k;
	int min_val_ij = 0; // Renamed 'min' to 'min_val_ij' to avoid conflict with MIN macro.

	printf("NEOPT SOLVER\n");
	// Block Logic: Allocates memory for temporary matrix C (result of A*B_upper_triangle).
	double *C = (double*) calloc(N * N, sizeof(double));
	if (C == NULL) {
		printf("Failed calloc for C!");
		return NULL;
	}

	double *At, *Bt; // Pointers for transposed matrices.

	// Block Logic: Allocates memory for another temporary matrix another_C (result of C * B_transpose).
	double *another_C = (double*) calloc(N * N, sizeof(double));
	if (another_C == NULL) {
		printf("Failed calloc for another_C!");
		free(C);
		return NULL;
	}

	// Block Logic: Allocates memory for temporary matrix res_A (result of A_transpose * A_lower_triangle).
	double *res_A = (double*) calloc(N * N, sizeof(double));
	if (res_A == NULL) {
		printf("Failed calloc for res_A!");
		free(C);
		free(another_C);
		return NULL;
	}

	// Block Logic: Allocates memory for the final result matrix.
	double *res = (double*) calloc(N * N, sizeof(double));
	if (res == NULL) {
		printf("Failed calloc for res!");
		free(C);
		free(another_C);
		free(res_A);
		return NULL;
	}

	// Block Logic: Computes the transpose of matrix A.
	At = my_transpose(N, A);
	// Block Logic: Computes the transpose of matrix B.
	Bt = my_transpose(N, B);

	// Block Logic: Performs a partial matrix multiplication (A * B_upper_triangle) storing result in C.
	// This loop structure suggests that B is treated as an upper triangular matrix or
	// that a specific part of the product is being computed.
	// Invariant: C[i][j] accumulates sum of A[i][k] * B[k][j] for k from i to N-1.
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			C[i * N + j] = 0.0; // Inline: Initializes current element of C to zero.
			for (k = i; k < N; ++k) { // Functional Utility: Inner loop for matrix multiplication.
				C[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	// Block Logic: Performs matrix multiplication (C * B_transpose) storing result in another_C.
	// Invariant: another_C[i][j] accumulates sum of C[i][k] * Bt[k][j].
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			another_C[i * N + j] = 0.0; // Inline: Initializes current element of another_C to zero.
			for (k = 0; k < N; ++k) { // Functional Utility: Inner loop for matrix multiplication.
				another_C[i * N + j] += C[i * N + k] * Bt[k * N + j];
			}
		}
	}

	// Block Logic: Performs a partial matrix multiplication (A_transpose * A_lower_triangle) storing result in res_A.
	// This loop structure suggests a specific lower triangular product.
	// Invariant: res_A[i][j] accumulates sum of At[i][k] * A[k][j] for k from 0 to min(i,j).
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			min_val_ij = MIN(i, j); // Inline: Determines the upper bound for k to compute a lower triangular part.
			res_A[i * N + j] = 0.0; // Inline: Initializes current element of res_A to zero.
			for(k = 0; k < min_val_ij + 1; ++k) { // Functional Utility: Inner loop for matrix multiplication.
				res_A[i * N + j] += At[i * N + k] * A[k * N + j];
			}
		}
	}

	// Block Logic: Performs matrix addition: res = another_C + res_A.
	// Invariant: res[i][j] = another_C[i][j] + res_A[i][j].
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			res[i * N + j] = another_C[i * N + j] + res_A[i * N + j];
		}
	}
	// Functional Utility: Frees all temporary dynamically allocated matrices.
	free(res_A);
	free(another_C);
	free(At);
	free(Bt);
	free(C);
	return res; // Functional Utility: Returns the pointer to the dynamically allocated final result matrix.
}
