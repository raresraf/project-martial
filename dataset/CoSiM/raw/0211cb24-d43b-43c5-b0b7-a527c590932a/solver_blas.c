
#include "utils.h"
#include "cblas.h"


double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");

	double *res1 = calloc(N * N, sizeof(double));
	if (res1 == NULL)
		exit(EXIT_FAILURE);

	double *res2 = calloc(N * N, sizeof(double));
	if (res2 == NULL)
		exit(EXIT_FAILURE);

	
	cblas_dcopy(N * N, B, 1, res1, 1);

	
	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper,
		CblasNoTrans, CblasNonUnit, N, N, 1, A, N, res1, N);

	
	cblas_dcopy(N * N, A, 1, res2, 1);

	
	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper,
		CblasTrans, CblasNonUnit, N, N, 1, A, N, res2, N);

	
	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
				N, N, N, 1, res1, N, B, N, 1, res2, N);


	free(res1);
	return res2;
}
