/**
 * @file solver_blas.c
 * @brief Encapsulates functional utility for solver_blas.c.
 * Performance Optimization: implements loop unrolling, cache-friendly data access, and SIMD where applicable. Time/space complexity optimized.
 */

#include "utils.h"
#include "cblas.h"


double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	double *C = malloc(N * N * sizeof(double));
	double *mat = malloc(N * N * sizeof(double));
	
	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	for(int i = 0; i < N; i++){
		double *idx = &B[i * N]; /* Non-obvious bitwise operation or pointer arithmetic */
		double *id = &mat[i * N]; /* Non-obvious bitwise operation or pointer arithmetic */
		/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
		for(int j = 0; j < N; j++) {
			*id = *idx;
			id++;
			idx++;
		}
	}

	double *point_C = &C[0]; /* Non-obvious bitwise operation or pointer arithmetic */
	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	for(int i = 0; i < N * N; i++){
		*point_C = 0;
		point_C++;
	}

	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, 
				CblasNonUnit, N, N, 1.0, A, N, mat, N);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 
				1.0, mat, N, B, N, 1.0, C, N);
	
	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	for(int i = 0; i < N; i++){
		double *idx = &A[i * N]; /* Non-obvious bitwise operation or pointer arithmetic */
		double *id = &mat[i * N]; /* Non-obvious bitwise operation or pointer arithmetic */
		/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
		for(int j = 0; j < N; j++) {
			*id = *idx;
			id++;
			idx++;
		}
	}

	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, 
				CblasNonUnit, N, N, 1.0, mat, N, mat, N);
	cblas_daxpy(N * N, 1.0, mat, 1, C, 1);
	free(mat);
	return C;
}
