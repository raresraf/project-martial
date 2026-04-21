/**
 * @78967c0c-ba90-43f5-87be-b35b30882925/solver_blas.c
 * @brief Optimized matrix expression solver utilizing Level 3 BLAS routines.
 *
 * Functional Utility: Solves the expression Result = (A * B) * B^T + (A^T * A)
 * using high-performance TRMM and GEMM kernels. This implementation leverages the 
 * upper triangular property of matrix A to reduce computational complexity.
 *
 * Domain: HPC Numerical Optimization.
 */

#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <cblas.h>


/**
 * @brief High-performance solver kernel utilizing BLAS routine delegation.
 * Logic:
 * 1. Multiplies B by A (triangular) in-place: B = A * B.
 * 2. Computes the Gramian A^T * A in-place within a copy of A (buffer C).
 * 3. Final aggregation: C = (B * B^T) + C.
 */
double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	
	double *Bt;
	double *C;
	
	// Pre-condition: Allocates workspace for the transpose of B and the result C.
	Bt = calloc(N * N, sizeof(double));
	C = calloc(N * N, sizeof(double));
	
	
	// Logic: Initializing C with A and Bt with B for subsequent multiplications.
	memcpy(C, A, N * N * sizeof(double));
	memcpy(Bt, B, N * N * sizeof(double));

	/**
	 * Block Logic: Compute B = A * B.
	 * Optimization: Uses `cblas_dtrmm` to exploit A's upper triangularity.
	 * Invariant: Destructive update to input matrix buffer B.
	 */
	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasNoTrans,
		CblasNonUnit,
		N, N,
		1.0, A, N,
		B, N
	);
	
	/**
	 * Block Logic: Compute C = A^T * A.
	 * Optimization: Efficiently computes the Gramian matrix of A in-place using TRMM.
	 */
	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasTrans,
		CblasNonUnit,
		N, N,
		1.0, C, N,
		C, N
	);
	
	/**
	 * Block Logic: Final summation stage: C = (B_new * Bt^T) + C.
	 * Functional Utility: General matrix multiplication with explicit transposition.
	 * Logic: Accumulates the product of modified B and the original B (Bt) into C.
	 */
	cblas_dgemm(
		CblasRowMajor,
		CblasNoTrans,
		CblasTrans,
		N, N, N,
		1.0, B, N,
		Bt, N,
		1.0, C, N
	);
	
	/**
	 * Cleanup: Releases auxiliary buffer Bt.
	 */
	free(Bt);
	
	return C;
}
