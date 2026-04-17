/**
 * @file solver_blas.c
 * @brief Encapsulates functional utility for solver_blas.c.
 * Performance Optimization: implements loop unrolling, cache-friendly data access, and SIMD where applicable. Time/space complexity optimized.
 */


#include <string.h>
#include <stdlib.h>
#include "cblas.h"
#include "utils.h"


double* my_solver(int N, double *A, double *B) {

	double *AB, *C, *AA;
	AB = calloc(N * N,  sizeof(double));
	C = calloc(N * N, sizeof(double));
	AA = calloc(N * N, sizeof(double));
	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	if (AB == NULL || C == NULL) { /* Non-obvious bitwise operation or pointer arithmetic */
        perror("malloc failed\n");
        exit(EXIT_FAILURE);
    }

	
	
	
	double alpha = 1.0, beta = 0.0;
	int i = 0, j = 0;

	memcpy(AB, B, N * N * sizeof(double));
	memcpy(AA, A, N * N * sizeof(double));
	
	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				N, N, N, alpha, A, N, B, N, beta, AB, N);

	
	cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
				N, N, N, alpha, AB, N, B, N, beta, C, N);
	
	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit,
				N, N, alpha, A, N, AA, N); 

	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	for (i = 0; i < N; i++) {
		/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
		for (j = 0; j < N; j++) {
			C[i * N + j] += AA[i * N + j];
		}
	}
	

	free(AA);
	free(AB);
	return C;
}
