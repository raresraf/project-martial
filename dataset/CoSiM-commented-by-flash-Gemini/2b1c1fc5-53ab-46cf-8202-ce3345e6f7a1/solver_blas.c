/**
 * @file solver_blas.c
 * @brief Semantic documentation for solver_blas.c. 
 *        This is a placeholder. Detailed semantic analysis will be applied later.
 */

#include "utils.h"
#include <string.h>
#include <stdlib.h>
#include "cblas.h"


double* my_solver(int N, double *A, double *B) {
	double *C;
	double *AA;

	C = malloc(N * N * sizeof(*C));
	if (NULL == C)
		exit(EXIT_FAILURE);

	AA = malloc(N * N * sizeof(*AA));
	if (NULL == AA)
		exit(EXIT_FAILURE);

	
	
	memcpy(C, B, N * N * sizeof(*C));

	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasNoTrans,
		CblasNonUnit,
		N, N,
		1.0, A, N,
		C, N
	);

	
	memcpy(AA, A, N * N * sizeof(*C));

	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasTrans,
		CblasNonUnit,
		N, N,
		1.0, A, N,
		AA, N
	);

	
	cblas_dgemm(
		CblasRowMajor,
		CblasNoTrans,
		CblasTrans,
		N, N, N,
		1.0, C, N,
		B, N,
		1.0, AA, N
	);

	free(C);

	return AA;
}
