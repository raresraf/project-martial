/**
 * @75135121-c23f-45b4-98b9-962563fdff8b/solver_neopt.c
 * @brief Naive implementation of a matrix expression solver.
 *
 * Functional Utility: Implements the matrix expression Result = (A * B) * B^T + (A^T * A)
 * using standard triple-nested iterative loops. This version serves as a baseline 
 * for performance comparisons against optimized or BLAS-based implementations.
 *
 * Domain: HPC Numerical Baselines.
 */

#include "utils.h"


/**
 * @brief Utility to compute the transpose of a square matrix.
 * Logic: Simple $O(N^2)$ element-wise swap from row-major to column-major layout.
 */
double* compute_transpose(int N, double* M)
{
	double* res = (double*)malloc(N * N * sizeof(double));
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			res[j * N + i] = M[i * N + j];
		}
	}
	return res;
}

/**
 * @brief Computes the matrix expression via standard iterative loops.
 * Algorithm: Stage-based linear algebra calculation.
 * Time Complexity: $O(N^3)$ due to triple-nested multiplication loops.
 */
double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	// Pre-condition: Allocates heap memory for all intermediate and final buffers.
	double* res_AxB = (double*)malloc(N * N * sizeof(double));
	double* res_ABBt = (double*)malloc(N * N * sizeof(double));
	double* res_AtA = (double*)malloc(N * N * sizeof(double));
	double* res = (double*)malloc(N * N * sizeof(double));
	double* B_t = compute_transpose(N, B);
	double* A_t = compute_transpose(N, A);

	
	/**
	 * Block Logic: Compute stage 1: (A * B).
	 * Invariant: Exploits the upper triangular property of matrix A by starting 
	 * the inner k-loop at index i.
	 */
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			res_AxB[i * N + j] = 0;
			for (int k = i; k < N; k++) {
				res_AxB[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	
	/**
	 * Block Logic: Compute stage 2: (A * B) * B^T.
	 * Logic: Leverages the explicitly computed transpose `B_t` for contiguous memory 
	 * access in the dot-product inner loop.
	 */
	int P = 0;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			res_ABBt[i * N + j] = 0;
			for (int k = 0; k < N; k++) {
				res_ABBt[i * N + j] += res_AxB[i * N + k] * B_t[k * N + j];
			}
		}
	}

	/**
	 * Block Logic: Compute stage 3: (A^T * A).
	 * Invariant: Exploits the combined sparsity of upper triangular A and its transpose.
	 * The loop bound P = min(i, j) minimizes redundant zero-product accumulations.
	 */
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			res_AtA[i * N + j] = 0;
			if (i < j) {
				P = i;
			}
			else {
				P = j;
			}
			for (int k = 0; k <= P; k++) {
				res_AtA[i * N + j] += A_t[i * N + k] * A[k * N + j];
			}
		}
	}
	
	/**
	 * Block Logic: Final summation pass.
	 */
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			res[i * N + j] = 0;
			res[i * N + j] += res_ABBt[i * N + j] + res_AtA[i * N + j];
		}
	}
	free(A_t);
	free(B_t);
	free(res_AtA);
	free(res_ABBt);
	free(res_AxB);
	return res;
}
