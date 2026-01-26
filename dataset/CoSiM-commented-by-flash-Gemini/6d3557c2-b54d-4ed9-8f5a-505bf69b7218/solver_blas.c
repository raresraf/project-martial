
#include "utils.h"
#include "cblas.h"
#include <string.h>


double* my_solver(int N, double *A, double *B) {
	
	int i, j;

	
	double *C = calloc(N * N, sizeof(double));
	memcpy(C, B, N * N * sizeof(double));

	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1.0, A, N, C, N);
    

	double *D = calloc(N * N, sizeof(double));
	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, C, N, B, N, 0, D, N);

	double *E = calloc(N * N, sizeof(double));
	memcpy(E, A, N * N * sizeof(double));
	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1.0, E, N, E, N);

	
	for(i = 0; i < N; i++) {
        for(j = 0; j < N; j++) {
			*(D + i * N + j) = *(D + i * N + j) + *(E + i * N + j);
        }
    }

	free(C);
	free(E);
	return D;
}
