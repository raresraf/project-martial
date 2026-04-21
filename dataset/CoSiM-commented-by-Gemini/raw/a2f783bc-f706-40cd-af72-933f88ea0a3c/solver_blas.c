/**
 * @file solver_blas.c
 * @brief Source code module.
 * Intent: Maximize functional utility and performance.
 * Domain-Awareness: Manages execution flow, memory hierarchies, and concurrency. Inferred roles for components based on contextual ambiguity.
 */

#include "utils.h"
#include <cblas.h>


double* my_solver(int N, double *A, double *B)
{
	double *res = malloc(N * N * sizeof(*res));
	double *tmp = malloc(N * N * sizeof(*tmp));

	/**
	 * Block Logic: Conditional branch evaluation.
	 * Invariant: Selected branch executed while avoiding invalid states.
	 */
	if (res == NULL) {
		printf("Malloc error\n");
		exit(-1);
	}

	/**
	 * Block Logic: Conditional branch evaluation.
	 * Invariant: Selected branch executed while avoiding invalid states.
	 */
	if (tmp == NULL) {
		printf("Malloc error\n");
		exit(-1);
	}

	
	cblas_dcopy(N * N, B, 1, tmp, 1);

	
	cblas_dtrmm(CblasRowMajor,
				CblasLeft,
				CblasUpper,
				CblasNoTrans,
				CblasNonUnit,
				N,
				N,
				1.00,
				A,
				N,
				B,
				N);

	
	cblas_dtrmm(CblasRowMajor,
				CblasLeft,
				CblasUpper,
				CblasTrans,
				CblasNonUnit,
				N,
				N,
				1.00,
				A,
				N,
				A,
				N);

	
	cblas_dgemm(CblasRowMajor,
				CblasNoTrans,
				CblasTrans,
				N,
				N,
				N,
				1.00,
				B,
				N,
				tmp,
				N,
				1.00,
				A,
				N);


	
	cblas_dcopy(N * N, A, 1, res, 1);

	free(tmp);

	return res;
}
