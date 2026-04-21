
#include "utils.h"
#include "cblas.h"
#include <string.h>

/**
 * @file solver_blas.c
 * @brief Optimized matrix solver using industry-standard BLAS library calls.
 * 
 * Functional Intent: Computes the resulting matrix for the expression:
 * Result = (A * B) * B^T + (A^T * A)
 * where A is an upper triangular matrix.
 * 
 * Performance Optimization: Utilizes Level 3 BLAS routines for high-performance 
 * linear algebra. It specifically employs `cblas_dgemm` for general matrix 
 * products and `cblas_dtrmm` to exploit the triangular structure of A 
 * during the symmetric product calculation.
 * 
 * Domain: HPC, Linear Algebra, BLAS Optimization.
 */

/**
 * my_solver - Implementation of the matrix expression via BLAS orchestration.
 * 
 * Algorithm: Multi-stage transformation.
 * 1. Compute T1 = A * B using `cblas_dgemm` (General multiply).
 * 2. Compute P1 = T1 * B^T using `cblas_dgemm` (Transposed multiply).
 * 3. Compute P2 = A^T * A using in-place `cblas_dtrmm` on a copy of A.
 * 4. Aggregate P1 and P2 via element-wise addition into the result buffer.
 */
double* my_solver(int N, double *A, double *B) {
	int i, j;
	double *ABBt, *result, *AtA;

	// Logic: Workspace and result buffer allocation.
	ABBt = malloc(N * N * sizeof(double));
	AtA = malloc(N * N * sizeof(double));
	result = malloc(N * N * sizeof(double));

	// Pre-condition: Prepare operand for in-place triangular multiplication.
	memcpy(AtA, A, N * N * sizeof(double));

	/**
	 * Block Logic: Compute (A * B).
	 * Algorithm: DGEMM.
	 * Logic: Performs initial general multiplication of A and B.
	 */
	cblas_dgemm(
		CblasRowMajor, CblasNoTrans,
		CblasNoTrans, N, N, N, 1.0, A,
		 N, B, N, 0.0, result, N);

	/**
	 * Block Logic: Compute (A * B) * B^T.
	 * Algorithm: DGEMM with Transposition.
	 * Invariant: result stores (A * B), which is then multiplied by B^T.
	 */
	cblas_dgemm(
		CblasRowMajor, CblasNoTrans,
		CblasTrans, N, N, N, 1.0, result, 
		N, B, N, 0.0, ABBt, N);

	/**
	 * Block Logic: Compute (A^T * A).
	 * Algorithm: DTRMM.
	 * Optimization: Exploits upper triangularity of A to compute the symmetric product efficiently.
	 */
	cblas_dtrmm( CblasRowMajor,
		CblasLeft, CblasUpper,
		CblasTrans, CblasNonUnit,
		N, N, 1.0, A, N, AtA, N
	);

	/**
	 * Block Logic: Final summation pass.
	 * Logic: Combines the two principal terms into the final output matrix.
	 */
	for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            result[i * N + j]  = ABBt[i * N + j] + AtA[i * N + j];
        }
    }

    free(AtA);
	free(ABBt);
	return result;
}
