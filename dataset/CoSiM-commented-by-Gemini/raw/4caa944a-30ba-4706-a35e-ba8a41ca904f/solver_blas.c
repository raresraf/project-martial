/**
 * @file solver_blas.c
 * @brief Encapsulates functional utility for solver_blas.c.
 * Performance Optimization: implements loop unrolling, cache-friendly data access, and SIMD where applicable. Time/space complexity optimized.
 */

/* @raw/4caa944a-30ba-4706-a35e-ba8a41ca904f/solver_blas.c: BLAS matrix solver */
#include "utils.h"
#include <cblas.h>
#include <string.h>


double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");

	double *C, *AtA, *AB;

	C = calloc(N * N, sizeof(double));
	DIE(C == NULL, "calloc C");

	AtA = calloc(N * N, sizeof(double));
	DIE(AtA == NULL, "calloc AtA");

	AB = calloc(N * N, sizeof(double));
	DIE(AB == NULL, "calloc AB");

	
	cblas_dcopy(N * N, B, 1, AB, 1);
	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasNoTrans,
		CblasNonUnit,
		N, N,
		1.0,
		A, N,
		AB, N
	);

	
	cblas_dcopy(N * N, A, 1, AtA, 1);
	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasTrans,
		CblasNonUnit,
		N, N,
		1.0,
		A, N,
		AtA, N
	);

	
	cblas_dcopy(N * N, AtA, 1, C, 1);
	cblas_dgemm(
		CblasRowMajor,
		CblasNoTrans,
		CblasTrans,
        N, N, N,
		1.00,
		AB, N,
		B, N,
		1.00,
		C, N);

	free(AB);
    free(AtA);

	return C;
}
