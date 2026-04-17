/**
 * @raw/1a3ac74d-7a0d-48f6-8b61-c6447bbc6dbc/solver_blas.c
 * @brief Computes the matrix expression C = A * B * B^T + A^T * A using highly optimized BLAS routines.
 * * Algorithm: Matrix multiplication with BLAS. Exploits the upper triangular property of matrix A.
 * Time Complexity: $O(N^3)$ dominated by the general matrix-matrix multiplication (GEMM).
 * Space Complexity: $O(N^2)$ for storing the result and intermediate matrices.
 */

#include <stdlib.h>
#include "utils.h"
#include "cblas.h"

/**
 * Functional Utility: Calculates the resulting matrix leveraging BLAS library operations.
 */
double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	
	double *C = (double *) malloc(N * N * sizeof(double));
	double *BBt = (double *) malloc(N * N * sizeof(double));

	/**
	 * Functional Utility: Computes BBt = B * B^T using general matrix multiplication.
	 */
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1, B, N, B, N, 0, BBt, N);

	/**
	 * Functional Utility: Computes A * BBt in-place into BBt using triangular matrix multiplication.
	 * Exploits the fact that matrix A is upper triangular.
	 */
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1, A, N, BBt, N);

	/**
	 * Functional Utility: Computes A^T * A in-place into A.
	 * Exploits the upper triangular nature of A.
	 */
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1, A, N, A, N);
	
	int i;
	/**
	 * Block Logic: Accumulates the final result C = (A * B * B^T) + (A^T * A).
	 */
	for (i = 0; i < N * N; ++i) {
		C[i] = BBt[i] + A[i];
	}

	return C;
}
