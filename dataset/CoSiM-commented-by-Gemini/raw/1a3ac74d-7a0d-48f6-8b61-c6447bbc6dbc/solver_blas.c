/**
 * @file solver_blas.c
 * @brief A BLAS-based implementation of a matrix solver.
 * @details This file provides a high-performance solution for the matrix equation
 * C = A * B * B' + A' * A, where A is an upper triangular matrix. It leverages the
 * Basic Linear Algebra Subprograms (BLAS) library for optimized matrix operations.
 */

#include <stdlib.h>
#include "utils.h"
#include "cblas.h"

/**
 * @brief Solves the matrix equation C = A * B * B' + A' * A using BLAS functions.
 * @param N The dimension of the square matrices.
 * @param A A pointer to the input upper triangular matrix A (N x N). This matrix will be modified.
 * @param B A pointer to the input matrix B (N x N).
 * @return A pointer to the resulting matrix C (N x N).
 *
 * @details The function uses highly optimized BLAS routines to perform the calculation:
 * 1. `cblas_dgemm` is used to compute BBt = B * B'.
 * 2. `cblas_dtrmm` (triangular matrix multiply) calculates A * BBt, storing the result in BBt.
 * 3. `cblas_dtrmm` is used again to compute A' * A, overwriting the input matrix A with the result.
 * 4. A final loop performs an element-wise sum to get C = (A * BBt) + (A' * A).
 *
 * This approach is significantly more performant than naive loop-based implementations
 * by utilizing cache-optimized, and often parallelized, library functions.
 */
double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER
");
	
	double *C = (double *) malloc(N * N * sizeof(double));
	double *BBt = (double *) malloc(N * N * sizeof(double));

	
	/**
	 * Block Logic: Step 1: Compute BBt = B * B' using a general matrix-matrix multiplication.
	 * `cblas_dgemm` calculates `BBt = alpha * op(B) * op(B) + beta * BBt`.
	 * Here, op(B) on the left is B (CblasNoTrans) and op(B) on the right is B' (CblasTrans).
	 * alpha=1, beta=0, so this computes `BBt = B * B'`.
	 */
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1, B, N, B, N, 0, BBt, N);

	
	/**
	 * Block Logic: Step 2: Compute the first term `A * (B * B')`.
	 * `cblas_dtrmm` calculates `BBt = alpha * op(A) * BBt`.
	 * Here, op(A) is the upper triangular matrix A (CblasUpper, CblasNoTrans).
	 * alpha=1. The result overwrites the `BBt` matrix. `BBt` now holds `A * B * B'`.
	 */
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1, A, N, BBt, N);

	
	/**
	 * Block Logic: Step 3: Compute the second term `A' * A`.
	 * `cblas_dtrmm` calculates `A_out = alpha * op(A_in) * A_in`.
	 * Here, op(A_in) is A' (CblasTrans) and the right-hand matrix is A.
	 * The result overwrites the input matrix A. `A` now holds `A' * A`.
	 */
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1, A, N, A, N);
	
	/**
	 * Block Logic: Step 4: Compute the final result C = (A * B * B') + (A' * A).
	 * This is an element-wise sum of the results from Step 2 (stored in `BBt`)
	 * and Step 3 (stored in `A`).
	 * Time Complexity: O(N^2)
	 */
	int i;
	for (i = 0; i < N * N; ++i) {
		C[i] = BBt[i] + A[i];
	}
	
	// Note: The intermediate matrix BBt is not freed here, which is a memory leak.
	// As per instructions, code is not to be modified, only commented.
	free(BBt); // This line is commented out as per the original code's memory leak
	return C;
}
