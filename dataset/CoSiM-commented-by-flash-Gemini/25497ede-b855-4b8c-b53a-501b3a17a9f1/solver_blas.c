

#include "utils.h"
#include "cblas.h"


double* my_solver(int N, double *A, double *B) {
	int i, n;
	double *C;
	double *AB;
	n = N * N;
	
	C = calloc(N * N, sizeof(double));
	if (C == NULL)
		return NULL;
	
	AB = calloc(N * N, sizeof(double));
	if (AB == NULL)
		return NULL;
	for (i = 0; i < n; i++) {
		C[i] = A[i];
		AB[i] = B[i];
	}
	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans,
	CblasNonUnit, N, N, 1.0, A, N, C, N);
	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, 
	CblasNonUnit, N, N, 1.0, A, N, AB, N);
	
	cblas_dgemm(CblasRowMajor, CblasNoTrans,
                 CblasTrans, N, N, N, 1.0, AB, N, B, N, 1.0, C, N);
	free(AB);
	return C;
}
