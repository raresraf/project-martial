/**
 * @file solver_blas.c
 * @brief High-level source code module.
 * Ensures cache-friendly data access, potential loop unrolling, and SIMD optimizations for C/C++.
 */

#include "utils.h"
#include <string.h>
#include <stdlib.h>
#include "cblas.h"
#include <math.h>
#include <stddef.h>

double* my_solver(int N, double *A, double *B)
{
	
	double *AtA, *C, *ABBt;

	
	C = malloc(N * N * sizeof(*C));
	/**
	 * @brief Pre-condition: Evaluates logical divergence based on current state.
	 * Invariant: Guarantees correct execution flow according to conditional partitioning.
	 */
	if (NULL == C)
		exit(1);

	AtA = malloc(N * N * sizeof(*AtA));
	/**
	 * @brief Pre-condition: Evaluates logical divergence based on current state.
	 * Invariant: Guarantees correct execution flow according to conditional partitioning.
	 */
	if (NULL == AtA)
		exit(1);
	ABBt = malloc(N * N * sizeof(*ABBt));
	/**
	 * @brief Pre-condition: Evaluates logical divergence based on current state.
	 * Invariant: Guarantees correct execution flow according to conditional partitioning.
	 */
	if (NULL == ABBt)
		exit(1);
	
	memcpy(C, B, N * N * sizeof(*C));

	
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
	 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
	 * Invariant: Operations within the block strictly maintain target functional boundaries.
	 */
	for (int i = 0; i < N; i++) 
		/**
		 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
		 * Invariant: Operations within the block strictly maintain target functional boundaries.
		 */
		for (int j = 0; j < N; j++)
			ABBt[i*N+j] += AtA[i*N+j];

	free(AtA);
	free(C);	
	return ABBt;
}
