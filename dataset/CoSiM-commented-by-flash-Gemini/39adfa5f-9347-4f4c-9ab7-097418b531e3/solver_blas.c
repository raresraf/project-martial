
#include "utils.h"
#include "cblas.h"
#include <string.h>


double* my_solver(int N, double *A, double *B) {
    double *partial_result1 = (double *)malloc(N * N * sizeof(double));
    double *partial_result2 = (double *)malloc(N * N * sizeof(double));

    
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
		N, N, N, 1.0, B, N, B, N, 0, partial_result1, N);

    
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
		N, N, 1.0, A, N, partial_result1, N);

    
    memcpy(partial_result2, A, N * N * sizeof(double));
    cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit,
		N, N, 1.0, A, N, partial_result2, N);

    
    cblas_daxpy(N * N, 1.0, partial_result1, 1, partial_result2, 1);

    free(partial_result1);
	return partial_result2;
}
