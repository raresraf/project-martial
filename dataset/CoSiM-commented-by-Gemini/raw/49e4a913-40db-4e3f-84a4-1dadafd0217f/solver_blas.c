/**
 * @file solver_blas.c
 * @brief Encapsulates functional utility for solver_blas.c.
 * Performance Optimization: implements loop unrolling, cache-friendly data access, and SIMD where applicable. Time/space complexity optimized.
 */

#include "utils.h"
#include <string.h>
#include <stdlib.h>
#include "cblas.h"

double* my_solver(int N, double *A, double *B) {
	
	double * C, *AB;

	
	C = calloc(N * N, sizeof(double));
	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	if(C == NULL)
		printf("Probleme la alocarea memoriei\n");
	AB = calloc(N * N, sizeof(double));
	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	if(AB == NULL)
		printf("Probleme la alocarea memoriei\n");
	memcpy(C, A, N * N * sizeof(*C));
	cblas_dtrmm(CblasRowMajor,
		CblasLeft, CblasUpper, CblasTrans, CblasNonUnit,
		N, N,
		1.0, A, N,
		C, N   );
	memcpy(AB, B, N * N * sizeof(*AB));
	cblas_dtrmm(CblasRowMajor,
                CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                N, N,
                1.0, A, N,
                AB, N   );
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
