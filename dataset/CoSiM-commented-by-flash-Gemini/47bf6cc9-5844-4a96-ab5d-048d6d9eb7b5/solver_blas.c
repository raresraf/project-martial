
#include <string.h>
#include <stdlib.h>
#include "cblas.h"
#include "utils.h"

/**
 * @file solver_blas.c
 * @brief Optimized matrix solver implementation using industry-standard BLAS calls.
 * 
 * Functional Intent: Computes the resulting matrix for the expression:
 * Result = (A * B) * B^T + (A^T * A)
 * where A is an upper triangular matrix.
 * 
 * Performance Optimization: Leverages Level 3 BLAS routines to achieve optimal 
 * throughput on high-dimensional data. It utilizes `cblas_dgemm` for general 
 * matrix multiplication and transposition, and `cblas_dtrmm` to exploit 
 * A's triangular properties during the symmetric product calculation.
 * 
 * Domain: HPC, Linear Algebra, BLAS Optimization.
 */

/**
 * my_solver - Implementation of the matrix expression via BLAS orchestration.
 * 
 * Algorithm: Atomic algebraic computation steps.
 * 1. Compute T1 = A * B using `cblas_dgemm` (General multiply).
 * 2. Compute P1 = T1 * B^T using `cblas_dgemm` (Transposed multiply).
 * 3. Compute P2 = A^T * A using in-place `cblas_dtrmm` on a copy of A.
 * 4. Aggregate P1 and P2 using element-wise addition.
 */
double* my_solver(int N, double *A, double *B) {
	double *AB, *C, *AA;

	// Logic: Workspace allocation for intermediate and component matrices.
	AB = calloc(N * N,  sizeof(double));
	C = calloc(N * N, sizeof(double));
	AA = calloc(N * N, sizeof(double));

	// Guard: Ensure memory availability before computation.
	if (AB == NULL || C == NULL || AA == NULL) {
        perror("malloc failed\n");
        exit(EXIT_FAILURE);
    }

	double alpha = 1.0, beta = 0.0;
	int i = 0, j = 0;

	/**
	 * Block Logic: Initial data setup.
	 */
	memcpy(AB, B, N * N * sizeof(double));
	memcpy(AA, A, N * N * sizeof(double));
	
	/**
	 * Block Logic: Compute (A * B).
	 * Algorithm: DGEMM.
	 * Logic: Performs initial general multiplication between operands A and B.
	 */
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				N, N, N, alpha, A, N, B, N, beta, AB, N);

	/**
	 * Block Logic: Compute (A * B) * B^T.
	 * Algorithm: DGEMM with Transposition.
	 * Invariant: 'AB' is transformed into the product with B^T, stored in 'C'.
	 */
	cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
				N, N, N, alpha, AB, N, B, N, beta, C, N);
	
	/**
	 * Block Logic: Compute (A^T * A).
	 * Algorithm: Triangular-aware DTRMM.
	 * Optimization: Exploits the upper triangular nature of A to compute the 
	 * Gramian matrix efficiently in-place on 'AA'.
	 */
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit,
				N, N, alpha, A, N, AA, N); 

	/**
	 * Block Logic: Final accumulation Result = C + AA.
	 * Logic: Naive element-wise summation to combine the two major terms.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i * N + j] += AA[i * N + j];
		}
	}
	
	free(AA);
	free(AB);
	return C;
}
