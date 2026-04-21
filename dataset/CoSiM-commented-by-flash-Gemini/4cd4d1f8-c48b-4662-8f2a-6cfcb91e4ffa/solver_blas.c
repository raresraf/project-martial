
#include "utils.h"
#include <string.h>
#include <stdlib.h>
#include "cblas.h"
#include <math.h>
#include <stddef.h>

/**
 * @file solver_blas.c
 * @brief Matrix expression solver utilizing standardized BLAS Level 3 operations.
 * 
 * Functional Intent: Computes the resulting matrix for the expression:
 * Result = (A * B) * B^T + (A^T * A)
 * where A is an upper triangular matrix.
 * 
 * Performance Optimization: Leverages standard BLAS library routines to 
 * optimize for hardware throughput. It specifically uses `cblas_dtrmm` to 
 * exploit the triangular properties of A and `cblas_dgemm` for efficient 
 * general matrix multiplication with transposition.
 * 
 * Domain: HPC, Linear Algebra, BLAS Optimization.
 */

/**
 * my_solver - Computes the complex matrix expression via sequential BLAS orchestration.
 * 
 * Algorithm: Atomic algebraic computation steps.
 * 1. Compute T1 = A * B using in-place `cblas_dtrmm` on a copy of B.
 * 2. Compute P1 = T1 * B^T using `cblas_dgemm` with transposition.
 * 3. Compute P2 = A^T * A using in-place `cblas_dtrmm` (Transposed multiply) on a copy of A.
 * 4. Aggregate P1 and P2 using element-wise addition.
 */
double* my_solver(int N, double *A, double *B)
{
	double *AtA, *C, *ABBt;

	// Logic: Allocation of result and workspace matrices.
	C = malloc(N * N * sizeof(*C));
	if (NULL == C)
		exit(1);

	AtA = malloc(N * N * sizeof(*AtA));
	if (NULL == AtA)
		exit(1);

	ABBt = malloc(N * N * sizeof(*ABBt));
	if (NULL == ABBt)
		exit(1);
	
	// Pre-condition: Initialize intermediate buffer for product T1.
	memcpy(C, B, N * N * sizeof(*C));

	/**
	 * Block Logic: Compute (A * B).
	 * Algorithm: DTRMM.
	 * Optimization: Uses triangular properties of A to multiply B directly in workspace C.
	 */
	cblas_dtrmm(CblasRowMajor, 
				CblasLeft,
                CblasUpper, 
                CblasNoTrans,
                CblasNonUnit, 
                N, 
                N,
                1.0, 
                A, 
                N,
                C, 
                N
                );
	
	/**
	 * Block Logic: Compute (A * B) * B^T.
	 * Algorithm: DGEMM with transposition.
	 * Invariant: 'ABBt' stores the result of multiplying intermediate 'C' by B^T.
	 */
	memcpy(ABBt, B, N * N * sizeof(*C));
	cblas_dgemm(CblasRowMajor, 
				CblasNoTrans,
                CblasTrans,
                N,
                N,
                N,
                1.0,
                C,
                N,
                B,
                N,
                0.0,
                ABBt,
                N
                );

	/**
	 * Block Logic: Compute (A^T * A).
	 * Algorithm: Triangular-aware DTRMM.
	 * Logic: Computes the symmetric product of triangular A directly into workspace AtA.
	 */
	memcpy(AtA, A, N * N * sizeof(*C));
	cblas_dtrmm(CblasRowMajor, 
				CblasLeft,
                CblasUpper, 
                CblasTrans,
                CblasNonUnit, 
                N, 
                N,
                1.0, 
                A, 
                N,
                AtA, 
                N
                );

	/**
	 * Block Logic: Final accumulation Result = ABBt + AtA.
	 */
	for (int i = 0; i < N; i++) 
		for (int j = 0; j < N; j++)
			ABBt[i*N+j] += AtA[i*N+j];

	free(AtA);
	free(C);	
	return ABBt;
}
