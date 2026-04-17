/**
 * @file solver_opt.c
 * @brief An optimized implementation of a matrix solver.
 * @details This file provides a micro-optimized solution for the matrix equation
 * C = A * B * B' + A' * A, where A is an upper triangular matrix. It builds upon
 * the baseline version by introducing register-based optimizations.
 */

#include <stdlib.h>
#include "utils.h"


/**
 * @brief Computes the transpose of a square matrix.
 * @param N The dimension of the square matrix.
 * @param A The input matrix (N x N) to be transposed.
 * @param B The output matrix (N x N) where the transpose will be stored.
 *
 * This function performs an out-of-place matrix transposition.
 * Time Complexity: O(N^2)
 */
void transpose(int N, double *A, double *B) {
	int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
			B[i * N + j] = A[j * N + i];
		}
	}
}

/**
 * @brief Solves the matrix equation C = A * B * B' + A' * A using a micro-optimized approach.
 * @param N The dimension of the square matrices.
 * @param A A pointer to the input upper triangular matrix A (N x N).
 * @param B A pointer to the input matrix B (N x N).
 * @return A pointer to the resulting matrix C (N x N).
 *
 * @details This version is functionally identical to the non-optimized solver but uses
 * the `register` keyword as a hint to the compiler to store frequently accessed variables
 * in CPU registers. It also introduces a local `sum` variable within the loops to
 * potentially improve data locality and reduce memory access.
 *
 * 1. Transposes B to get B'.
 * 2. Calculates BBt = B * B'.
 * 3. Calculates ABBt = A * BBt, exploiting the upper triangular nature of A.
 * 4. Transposes A to get A'.
 * 5. Calculates AtA = A' * A, exploiting the triangular nature of A and A'.
 * 6. Computes the final result C = ABBt + AtA.
 */
double* my_solver(int N, double *A, double *B) {
	printf("OPT SOLVER
");
	// Use 'register' as a hint to the compiler for frequently accessed pointers.
	register double *C = (double *) malloc(N * N * sizeof(double));
	
	register double *ABBt = (double *) malloc(N * N * sizeof(double));
	register double *At = (double *) malloc(N * N * sizeof(double));
	register double *AtA = (double *) malloc(N * N * sizeof(double));

	register double *BBt = (double *) malloc(N * N * sizeof(double));
	register double *Bt = (double *) malloc(N * N * sizeof(double));

	transpose(N, B, Bt);
	register int i, j, k;

	
	/**
	 * Block Logic: Compute BBt = B * B' with register-based accumulators.
	 * A local 'sum' variable is used for accumulation to encourage register usage.
	 * Time Complexity: O(N^3)
	 */
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			register double sum = 0.0;
			for (k = 0; k < N; ++k) {
				sum += B[i * N + k] * Bt[k * N + j];
			}
			BBt[i * N + j] = sum;
		}
	}

	
	/**
	 * Block Logic: Compute ABBt = A * BBt.
	 * Exploits the upper triangular property of A (k starts from i).
	 * Time Complexity: O(N^3), with a reduced operation count.
	 */
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			register double sum = 0.0;
			for (k = i; k < N; ++k) {
				sum += A[i * N + k] * BBt[k * N + j];
			}
			ABBt[i * N + j] = sum;
		}
	}

	transpose(N, A, At);

	
	/**
	 * Block Logic: Compute AtA = A' * A.
	 * Exploits the lower triangular property of A' (k <= i) and the upper
	 * triangular property of A (k <= j).
	 * Time Complexity: O(N^3), with a reduced operation count.
	 */
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			register double sum = 0.0;
			for (k = 0; k <= i && k <= j; ++k) {
				sum += At[i * N + k] * A[k * N + j];
			}
			AtA[i * N + j] = sum;
		}
	}

	/**
	 * Block Logic: Final element-wise addition C = ABBt + AtA.
	 * Time Complexity: O(N^2)
	 */
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			C[i * N + j] = ABBt[i * N + j] + AtA[i * N + j]; 
		}
	}
	
	// Free all dynamically allocated intermediate matrices.
	free(ABBt);
	free(At);
	free(AtA);
	free(BBt);
	free(Bt);
	return C;
}
