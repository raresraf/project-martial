
/**
 * @file solver_neopt.c
 * @brief A non-optimized, modular implementation of a matrix solver.
 *
 * This file provides a clean, textbook-style C implementation for a series of
 * matrix operations. It is well-structured with distinct helper functions
 * for each operation (e.g., upper/lower triangular multiplication, transpose, add),
 * making the logic easy to follow.
 */
#include "utils.h"

/**
 * @brief Multiplies an upper triangular matrix with a general matrix (C += A * B).
 * @param N The dimension of the matrices.
 * @param A The upper triangular input matrix.
 * @param B The second input matrix.
 * @param C The output matrix where the result is accumulated.
 */
void matrix_mul_upper(int N, double *A, double *B, double *C)
{
	int i, j, k;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			// Inner loop starts from k=i, assuming A is upper triangular.
			for (k = i; k < N; k++) {
				C[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}
}

/**
 * @brief Multiplies a lower triangular matrix with a general matrix (C += A * B).
 * @param N The dimension of the matrices.
 * @param A The lower triangular input matrix.
 * @param B The second input matrix.
 * @param C The output matrix where the result is accumulated.
 */
void matrix_mul_lower(int N, double *A, double *B, double *C)
{
	int i, j, k;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			// Inner loop ends at k=i, assuming A is lower triangular.
			for (k = 0; k <= i; k++) {
				C[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}
}

/**
 * @brief Performs a naive, general matrix-matrix multiplication (C += A * B).
 * @param N The dimension of the matrices.
 * @param A The first input matrix.
 * @param B The second input matrix.
 * @param C The output matrix where the result is accumulated.
 */
void matrix_mul(int N, double *A, double *B, double *C)
{
	int i, j, k;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				C[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}
}

/**
 * @brief Computes the transpose of a square matrix (AT = A^T).
 * @param N The dimension of the matrix.
 * @param A The input matrix.
 * @param AT The output matrix to store the transpose.
 */
void matrix_transpose(int N, double *A, double *AT)
{
	int i, j;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			AT[i * N + j] = A[j * N + i];
		}
	}
}

/**
 * @brief Adds two matrices (C = A + B).
 * @param N The dimension of the matrices.
 * @param A The first input matrix.
 * @param B The second input matrix.
 * @param C The output matrix.
 */
void matrix_add(int N, double *A, double *B, double *C)
{
	int i, j;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i * N + j] = A[i * N + j] + B[i * N + j];
		}
	}
}

/**
 * @brief Performs a sequence of matrix operations using modular helper functions.
 *
 * @param N The dimension of the square matrices A and B.
 * @param A A pointer to the first input matrix (N x N).
 * @param B A pointer to the second input matrix (N x N).
 * @return A pointer to the resulting matrix.
 *
 * @note This function computes the expression: C = (A * B) * B^T + A^T * A.
 *       It explicitly breaks down the computation into clear, sequential steps
 *       using dedicated helper functions for each operation.
 */
double* my_solver(int N, double *A, double* B)
{
	
	// Step 1: Compute AB = A * B, treating A as upper triangular.
	double *AB = calloc(N * N, sizeof(double));
	matrix_mul_upper(N, A, B, AB);

	
	// Step 2: Compute the transpose of B.
	double *B_T = malloc(N * N * sizeof(double));
	matrix_transpose(N, B, B_T);

	
	// Step 3: Compute ABB = (A * B) * B^T.
	double *ABB = calloc(N * N, sizeof(double));
	matrix_mul(N, AB, B_T, ABB);

	
	// Step 4: Compute the transpose of A.
	double *A_T = malloc(N * N * sizeof(double));
	matrix_transpose(N, A, A_T);

	
	// Step 5: Compute AA = A^T * A.
	// Since A is upper triangular, A^T is lower triangular, so we use matrix_mul_lower.
	double *AA = calloc(N * N, sizeof(double));
	matrix_mul_lower(N, A_T, A, AA);

	
	// Step 6: Perform the final addition C = ABB + AA.
	double *C = malloc(N * N * sizeof(double));
	matrix_add(N, ABB, AA, C);

	
	// Cleanup intermediate allocations.
	free(A_T);
	free(AB);
	free(B_T);
	free(ABB);
	free(AA);

	return C;
}
