
#include "utils.h"
#include "cblas.h"

#define ONE 1.0


double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");

	double *C, *AB, *AA;

	C = (double *)calloc(sizeof(double), N * N);
	AB = (double *)calloc(sizeof(double), N * N);
	AA = (double *)calloc(sizeof(double), N * N);

	cblas_dcopy(N * N, A, 1, AA, 1);

	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit,
		N, N, ONE, A, N, AA, N);

	cblas_dcopy(N * N, B, 1, AB, 1);

	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans,
		CblasNonUnit, N, N, ONE, A, N, AB, N);

	cblas_dcopy(N * N, AA, 1, C, 1);

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, ONE, AB, N, B,
		N, ONE, C, N);

	free(AB);
	free(AA);
	return C;
}
