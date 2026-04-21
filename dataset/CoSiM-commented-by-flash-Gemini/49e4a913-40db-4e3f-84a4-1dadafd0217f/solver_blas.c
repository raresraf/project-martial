
#include "utils.h"
#include <string.h>
#include <stdlib.h>
#include "cblas.h"

/**
 * @file solver_blas.c
 * @brief Optimized matrix solver implementation using standardized BLAS library calls.
 * 
 * Functional Intent: Computes the resulting matrix for the expression:
 * Result = (A^T * A) + (A * B) * B^T
 * where A is an upper triangular matrix.
 * 
 * Performance Optimization: Leverages standard Level 3 BLAS routines to 
 * optimize for hardware throughput. Specifically, it uses `cblas_dtrmm` for 
 * triangular matrix multiplication and `cblas_dgemm` for general matrix 
 * products with transposition and additive updates.
 * 
 * Domain: HPC, Linear Algebra, BLAS Optimization.
 */

/**
 * my_solver - Computes the complex matrix expression via sequential BLAS orchestration.
 * 
 * Algorithm: Atomic algebraic stages.
 * 1. Compute P1 = A^T * A using in-place `cblas_dtrmm` (Left, Upper, Trans) on a copy of A.
 * 2. Compute T1 = A * B using `cblas_dtrmm` (Left, Upper, NoTrans) on a copy of B.
 * 3. Compute Result = P1 + T1 * B^T using `cblas_dgemm` with transposition 
 *    and additive update (beta=1.0).
 */
double* my_solver(int N, double *A, double *B) {
	double * C, *AB;

	// Logic: Allocation and initialization of workspace and result matrices.
	C = calloc(N * N, sizeof(double));
	if(C == NULL)
		printf("Probleme la alocarea memoriei\n");
	
	AB = calloc(N * N, sizeof(double));
	if(AB == NULL)
		printf("Probleme la alocarea memoriei\n");

	/**
	 * Block Logic: Compute (A^T * A).
	 * Algorithm: In-place DTRMM.
	 * Logic: Transforms the upper triangular matrix A into the symmetric product 
	 * using triangular-aware multiplication.
	 */
	memcpy(C, A, N * N * sizeof(*C));
	cblas_dtrmm(CblasRowMajor,
		CblasLeft, CblasUpper, CblasTrans, CblasNonUnit,
		N, N,
		1.0, A, N,
		C, N   );

	/**
	 * Block Logic: Compute (A * B).
	 * Algorithm: DTRMM.
	 * Optimization: Exploits upper triangularity of A to transform a copy of B.
	 */
	memcpy(AB, B, N * N * sizeof(*AB));
	cblas_dtrmm(CblasRowMajor,
                CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                N, N,
                1.0, A, N,
                AB, N   );

	/**
	 * Block Logic: Final accumulation Result = C + (AB * B^T).
	 * Algorithm: DGEMM.
	 * Invariant: 'C' already contains the (A^T * A) term; the product of AB 
	 * and transposed B is added directly into it.
	 */
	cblas_dgemm(CblasRowMajor,
                CblasNoTrans,
		CblasTrans,
		N, N, N, 1.0,
		AB, N, B, N,
		1.0, C, N  
	);
	
	free(AB);
	return C;
}
