
#include "utils.h"
#include "cblas.h"
#include <string.h>


double* my_solver(int N, double *A, double *B) {
	double *Acopy = malloc(N * N * sizeof(double));
	double *Bcopy = malloc(N * N * sizeof(double));

	memcpy(Acopy, A, N * N * sizeof(double));
	memcpy(Bcopy, B, N * N * sizeof(double));

	
    	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans,
                CblasNonUnit, N, N, 1, A, N, Bcopy, N);

	
    	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans,
                CblasNonUnit, N, N, 1, A, N, Acopy, N);

	
    	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                N, N, N, 1, Bcopy, N, B, N, 1, Acopy, N);

	free(Bcopy);

	return Acopy;
}