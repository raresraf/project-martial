/**
 * @75e2b6c5-8d07-48c3-9608-9f3b2524dcda/solver_blas.c
 * @brief Optimized matrix expression solver using Level 3 BLAS.
 *
 * Functional Utility: Computes the matrix expression Result = (A * B * B^T) + (A^T * A)
 * by decomposing it into distinct high-performance linear algebra stages (GEMM and TRMM).
 *
 * Domain: HPC Numerical Optimization.
 */

#include <stdio.h>
#include <string.h>

#include "utils.h"
#include "cblas.h"


/**
 * @brief Solver kernel utilizing hardware-accelerated BLAS routines.
 * Logic:
 * 1. Computes the product B * B^T via `cblas_dgemm`.
 * 2. Multiplies the result by A via `cblas_dtrmm`, exploiting A's upper triangularity.
 * 3. Adds the Gramian matrix A^T * A using a final GEMM call with additive accumulation.
 */
double* my_solver(int N, double *A, double *B) {
	register int size = N * N * sizeof(double);

	
	/**
	 * Block Logic: Compute BB = B * B^T.
	 * Optimization: Standard dense GEMM with implicit transposition.
	 */
	double *BB = malloc(size);
	cblas_dgemm(
		CblasRowMajor,
		CblasNoTrans,
		CblasTrans,
		N, N, N,
		1, B, N,
		B, N, 0,
		BB, N
	);

	double *AB = malloc(size);
	memcpy(AB, BB, size);

	/**
	 * Block Logic: Compute AB = A * (B * B^T).
	 * Optimization: Uses `cblas_dtrmm` to take advantage of A being an upper triangular matrix, 
	 * significantly reducing FLOP count compared to a full GEMM.
	 */
	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasNoTrans,
		CblasNonUnit,
		N, N,
		1, A, N,
		AB, N
	);

	
	/**
	 * Block Logic: Final summation: C = AB + (A^T * A).
	 * Functional Utility: Computes the Gramian matrix and adds it directly to the 
	 * existing AB product (beta=1 parameter in DGEMM).
	 */
	double *C = malloc(size);
	memcpy(C, AB, size);

	cblas_dgemm(
		CblasRowMajor,
		CblasTrans,
		CblasNoTrans,
		N, N, N,
		1, A, N,
		A, N, 1,
		C, N
	);


	free(AB);
	free(BB);

	return C;
}
