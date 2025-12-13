
#include "utils.h"
#include <cblas.h>
#include <string.h>


double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");

	double *C1 = (double*) calloc(N * N, sizeof(double));
	double *C2 = (double*) calloc(N * N, sizeof(double));
	int i;
	
	for(i = 0; i < N * N; i++)
		C1[i] = B[i];

	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, 
				CblasNoTrans, CblasNonUnit, N, N, 1,
				A, N, C1, N);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                N, N, N, 1, C1, N, B, N, 0, C2, N);

	for(i = 0; i < N * N; i++)
		C1[i] = A[i];

	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper,
				CblasTrans, CblasNonUnit, N, N, 1,
				A, N, C1, N);

	cblas_daxpy(N * N, 1, C2, 1, C1, 1);

	free(C2);

	return C1;
}
