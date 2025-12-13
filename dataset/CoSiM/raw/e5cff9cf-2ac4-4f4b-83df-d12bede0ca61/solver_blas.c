
#include "utils.h"
#include "cblas.h"

double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	int size = N * N;
	double *Bc = malloc(size * sizeof(double));
	if (!Bc)
		return NULL;
	double *Ac = malloc(size * sizeof(double));
	if (!Ac)
		return NULL;

	cblas_dcopy(size, A, 1, Ac, 1);
	cblas_dcopy(size, B, 1, Bc, 1);

	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans,
				CblasNonUnit, N, N, 1, A, N, Bc, N);

	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans,
				CblasNonUnit, N, N, 1, A, N, Ac, N);

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
				N, N, N, 1, Bc, N, B, N, 1, Ac, N);

	free(Bc);
	return Ac;
}
