
/**
 * @file solver_opt.c
 * @brief An optimized, modular implementation of a matrix solver.
 *
 * This file contains an optimized version of the modular, non-optimized solver.
 * It applies classic C micro-optimizations, such as using the `register`
 * keyword and reordering loops to improve data locality and cache performance.
 */
#include "utils.h"

/**
 * @brief Multiplies an upper triangular matrix with a general matrix (C += A * B).
 *
 * This version is optimized by reordering the loops to an i, k, j order.
 * This improves data locality by reusing the value of A[i][k] for the entire
 * inner loop over j.
 *
 * @param N The dimension of the matrices.
 * @param A The upper triangular input matrix.
 * @param B The second input matrix.
 * @param C The output matrix where the result is accumulated.
 */
static void matrix_mul_upper(register int N, register double *A, register double *B, register double *C)
{
	register int i, j, k;
	for (i = 0; i < N; i++) {
		register int line = i * N;

		// Loop order is i, k, j.
		for (k = i; k < N; k++) {
			// Optimization: Load A[i][k] once and reuse it.
			register double pa = A[line + k];
			register int k_line = k * N;

			for (j = 0; j < N; j++) {
				C[line + j] += pa * B[k_line + j];
			}
		}
	}
}

/**
 * @brief Multiplies a lower triangular matrix with a general matrix (C += A * B).
 *
 * Optimized with an i, k, j loop order for better data locality.
 *
 * @param N The dimension of the matrices.
 * @param A The lower triangular input matrix.
 * @param B The second input matrix.
 * @param C The output matrix where the result is accumulated.
 */
static void matrix_mul_lower(register int N, register double *A, register double *B, register double *C)
{
	register int i, j, k;
	for (i = 0; i < N; i++) {
		register int line = i * N;

		for (k = 0; k <= i; k++) {
			register double pa = A[line + k];
			register int k_line = k * N;

			for (j = 0; j < N; j++) {
				C[line + j] += pa * B[k_line + j];
			}
		}
	}
}

/**
 * @brief Performs a general matrix-matrix multiplication (C += A * B).
 *
 * Optimized with an i, k, j loop order for better data locality.
 *
 * @param N The dimension of the matrices.
 * @param A The first input matrix.
 * @param B The second input matrix.
 * @param C The output matrix where the result is accumulated.
 */
static void matrix_mul(register int N, register double *A, register double *B, register double *C)
{
	register int i, j, k;
	for (i = 0; i < N; i++) {
		register int line = i * N;

		for (k = 0; k < N; k++) {
			register double pa = A[line + k];
			register int k_line = k * N;

			for (j = 0; j < N; j++) {
				C[line+ j] += pa * B[k_line + j];
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
static void matrix_transpose(register int N, register double *A, register double *AT)
{
	register int i, j;
	for (i = 0; i < N; i++) {
		register int line = i * N;

		for (j = 0; j < N; j++) {
			AT[line + j] = A[j * N + i];
		}
	}
}

/**
 * @brief Adds two matrices (C = A + B). This version flattens the loop for potential optimization.
 * @param N The dimension of the matrices.
 * @param A The first input matrix.
 * @param B The second input matrix.
 * @param C The output matrix.
 */
static void matrix_add(register int N, register double *A, register double *B, register double *C)
{
	register int i;
	for (i = 0; i < N * N; i++) {
		C[i] = A[i] + B[i];
	}
}


/**
 * @brief Performs a sequence of matrix operations using optimized modular helper functions.
 *
 * @param N The dimension of the square matrices A and B.
 * @param A A pointer to the first input matrix (N x N).
 * @param B A pointer to the second input matrix (N x N).
 * @return A pointer to the resulting matrix.
 *
 * @note This function computes the expression: C = (A * B) * B^T + A^T * A.
 *       It uses the same clean, step-by-step logic as the non-optimized modular
 *       version, but calls helper functions that have been manually optimized.
 */
double* my_solver(int N, double *A, double* B) {
	register int size = N * N * sizeof(double);
	
	// Step 1: Compute the transpose of A.
	register double *A_T = malloc(size);
	matrix_transpose(N, A, A_T);

	
	// Step 2: Compute AB = A * B, treating A as upper triangular.
	register double *AB = calloc(N * N, sizeof(double));
	matrix_mul_upper(N, A, B, AB);

	
	// Step 3: Compute the transpose of B.
	register double *B_T = malloc(size);
	matrix_transpose(N, B, B_T);

	
	// Step 4: Compute ABB = (A * B) * B^T.
	register double *ABB = calloc(N * N, sizeof(double));
	matrix_mul(N, AB, B_T, ABB);

	
	// Step 5: Compute AA = A^T * A.
	register double *AA = calloc(N * N, sizeof(double));
	matrix_mul_lower(N, A_T, A, AA);

	
	// Step 6: Perform the final addition C = ABB + AA.
	register double *C = malloc(size);
	matrix_add(N, ABB, AA, C);

	
	// Cleanup intermediate allocations.
	free(A_T);
	free(AB);
	free(B_T);
	free(ABB);
	free(AA);

	return C;
}
