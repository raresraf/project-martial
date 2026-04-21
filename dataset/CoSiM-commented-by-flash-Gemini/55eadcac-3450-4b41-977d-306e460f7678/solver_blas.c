/**
 * @file solver_blas.c
 * @brief High-performance matrix expression solver using Level 3 BLAS.
 *
 * This implementation solves the matrix expression:
 * Result = (A * B) * B^T + (A^T * A)
 * where A is an upper triangular matrix.
 *
 * Algorithm: Leverages cblas_dtrmm for triangular products and cblas_dgemm 
 * for general matrix multiplication with transposition.
 *
 * Optimization: Uses industry-standard BLAS library for highly efficient
 * memory access and vectorized computations.
 *
 * Domain: HPC, Linear Algebra.
 */

#include "utils.h"
#include <cblas.h>

/**
 * my_solver - Computes the matrix expression using BLAS routines.
 * @param N Dimension of the square matrices.
 * @param A Pointer to the upper triangular matrix A.
 * @param B Pointer to the general matrix B.
 * @return Pointer to the resulting matrix.
 */
double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	int i, j;

	/**
	 * Pre-condition: Allocation of temporary and result buffers.
	 */
	double* C = (double*)calloc(N * N, sizeof(double));
	double* prod1 = (double*)calloc(N * N, sizeof(double));
	double* result = (double*)calloc(N * N, sizeof(double));
	if (C == NULL || prod1 == NULL || result == NULL) {
        perror("calloc failed\n");
        exit(EXIT_FAILURE);
    }

	/**
	 * Invariant: Initializes C as a copy of B and result as a copy of A.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i * N + j] = B[i * N + j];
			result[i * N + j] = A[i * N + j];
		}
	}

	/**
	 * Block Logic: Compute C = A * B.
	 * Functional Utility: Exploits the upper triangular property of A.
	 */
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans,
				CblasNonUnit, N, N, 1.0, A, N, C, N);

	/**
	 * Block Logic: Compute result = A^T * A.
	 * Functional Utility: Efficiently computes the Gramian matrix for A.
	 */
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans,
				CblasNonUnit, N, N, 1.0, A, N, result, N);

	/**
	 * Block Logic: Compute result = 1.0 * (C * B^T) + 1.0 * result.
	 * Functional Utility: Performs the final multiplication and addition in one step.
	 */
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
				N, N, N, 1.0, C, N, B, N, 1.0, result, N);

	free(prod1);
	free(C);
	return result;
}
