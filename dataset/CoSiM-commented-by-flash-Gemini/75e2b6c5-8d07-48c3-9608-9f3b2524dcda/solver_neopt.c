/**
 * @75e2b6c5-8d07-48c3-9608-9f3b2524dcda/solver_neopt.c
 * @brief Naive implementation of a matrix expression solver.
 *
 * Functional Utility: Implements the matrix expression Result = (A * B * B^T) + (A^T * A)
 * using a series of unoptimized iterative kernels. Each stage of the computation 
 * is performed via a dedicated loop-based routine.
 *
 * Domain: HPC Numerical Baselines.
 */

#include "utils.h"

/**
 * @brief Performs matrix multiplication assuming the left operand is upper triangular.
 * Logic: Optimizes the inner dot-product loop by skipping zero-value entries.
 */
void matrix_mul_upper(int N, double *A, double *B, double *C)
{
	int i, j, k;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = i; k < N; k++) {
				C[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}
}

/**
 * @brief Performs matrix multiplication assuming the left operand is lower triangular.
 */
void matrix_mul_lower(int N, double *A, double *B, double *C)
{
	int i, j, k;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k <= i; k++) {
				C[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}
}

/**
 * @brief Standard dense matrix multiplication kernel.
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
 * @brief Performs an element-wise matrix transposition.
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
 * @brief Performs element-wise matrix addition.
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
 * @brief Orchestrates the multi-stage calculation of the target expression.
 * Algorithm: Sequential decomposition into transpose, multiply, and add stages.
 * Time Complexity: $O(N^3)$.
 */
double* my_solver(int N, double *A, double* B)
{
	
	// Stage 1: Compute AB = A * B (leveraging triangular A).
	double *AB = malloc(N * N * sizeof(double));
	matrix_mul_upper(N, A, B, AB);

	
	// Stage 2: Compute B^T.
	double *B_T = malloc(N * N * sizeof(double));
	matrix_transpose(N, B, B_T);

	
	// Stage 3: Compute ABB = AB * B^T.
	double *ABB = malloc(N * N * sizeof(double));
	matrix_mul(N, AB, B_T, ABB);

	
	// Stage 4: Compute A^T.
	double *A_T = malloc(N * N * sizeof(double));
	matrix_transpose(N, A, A_T);

	
	// Stage 5: Compute AA = A^T * A (leveraging triangular A^T).
	double *AA = malloc(N * N * sizeof(double));
	matrix_mul_lower(N, A_T, A, AA);

	
	// Stage 6: Consolidate terms into final matrix C.
	double *C = malloc(N * N * sizeof(double));
	matrix_add(N, ABB, AA, C);

	
	/**
	 * Cleanup: Deallocates all temporary matrix buffers.
	 */
	free(A_T);
	free(AB);
	free(B_T);
	free(ABB);
	free(AA);

	return C;
}
