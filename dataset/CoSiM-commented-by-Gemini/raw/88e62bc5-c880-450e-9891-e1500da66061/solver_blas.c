
#include "utils.h"
#include <string.h>
#include <cblas.h>


#define min(x, y) (((x) < (y)) ? (x) : (y))
double* my_solver(int N, double *A, double *B) {
	
	
	double *C = (double *)malloc(N * N * sizeof(double));
        double *term = (double *)malloc(N * N * sizeof(double));

	memcpy(term, B, N * N * sizeof(double));
	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1, A, N, term, N);	

	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1, A, N, A, N);

	memcpy(C, A, N * N * sizeof(double));
	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1, term, N, B, N, 1, C, N);

        free(term);

        return C;
}
                 
