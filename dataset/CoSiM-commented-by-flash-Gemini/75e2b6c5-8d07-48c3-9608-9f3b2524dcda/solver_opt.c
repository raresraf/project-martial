/**
 * @75e2b6c5-8d07-48c3-9608-9f3b2524dcda/solver_opt.c
 * @brief Manually optimized implementation of a matrix expression solver.
 *
 * Functional Utility: Computes the matrix expression Result = (A * B * B^T) + (A^T * A)
 * using a series of optimized iterative kernels. These kernels leverage loop reordering 
 * for cache friendliness and `register` hints for scalar promotion.
 *
 * Domain: HPC Performance Tuning.
 */

#include "utils.h"

/**
 * @brief Multiplies matrices with upper-triangular left operand using loop reordering.
 * Optimization: The (i, k, j) loop order ensures contiguous access to matrix B, 
 * improving cache line utilization and prefetching effectiveness.
 */
static void matrix_mul_upper(register int N, register double *A, register double *B, register double *C)
{
	register int i, j, k;
	for (i = 0; i < N; i++) {
		register int line = i * N;

		for (k = i; k < N; k++) {
			register double pa = A[line + k];
			register int k_line = k * N;

			for (j = 0; j < N; j++) {
				C[line + j] += pa * B[k_line + j];
			}
		}
	}
}

/**
 * @brief Multiplies matrices with lower-triangular left operand using cache-optimized layout.
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
 * @brief Standard dense matrix multiplication kernel with reordered loop indices.
 * Optimization: Uses a (k, j) inner loop structure to maintain spatial locality for B.
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
 * @brief Performs matrix transposition using manual indexing.
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
 * @brief Optimized element-wise matrix addition.
 * Optimization: Collapses two-dimensional iteration into a single linear pass 
 * to maximize memory throughput.
 */
static void matrix_add(register int N, register double *A, register double *B, register double *C)
{
	register int i;
	for (i = 0; i < N * N; i++) {
		C[i] = A[i] + B[i];
	}
}

/**
 * @brief High-level solver orchestrator using optimized sub-routines.
 * Optimization: Employs `register` hints for pointers and dimensions to minimize 
 * stack access and promote compiler scalar optimization.
 */
double* my_solver(int N, double *A, double* B) {
	register int size = N * N * sizeof(double);
	
	register double *A_T = malloc(size);
	matrix_transpose(N, A, A_T);

	
	register double *AB = malloc(size);
	matrix_mul_upper(N, A, B, AB);

	
	register double *B_T = malloc(size);
	matrix_transpose(N, B, B_T);

	
	register double *ABB = malloc(size);
	matrix_mul(N, AB, B_T, ABB);

	
	register double *AA = malloc(size);
	matrix_mul_lower(N, A_T, A, AA);

	
	register double *C = malloc(size);
	matrix_add(N, ABB, AA, C);

	
	free(A_T);
	free(AB);
	free(B_T);
	free(ABB);
	free(AA);

	return C;
}
