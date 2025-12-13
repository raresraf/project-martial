
#include <string.h>
#include <stdlib.h>
#include "cblas.h"
#include "utils.h"


double* my_solver(int N, double *A, double *B) {
	double *AB;
	double *first;
	double *second;
	double *C;
	AB = calloc(N * N, sizeof(double));
	if(AB == NULL){
		exit(-1);
	}
	first = calloc(N * N, sizeof(double));
	if(first == NULL){
		exit(-1);
	}
	second = calloc(N * N, sizeof(double));
	if(second == NULL){
		exit(-1);
	}
	C = calloc(N * N, sizeof(double));
	if(C == NULL){
		exit(-1);
	}
    
	cblas_dcopy(N * N, B, 1.0, AB, 1.0);
	cblas_dtrmm(CblasRowMajor, CblasLeft,CblasUpper,CblasNoTrans,CblasNonUnit, N, N, 1.0, A, N, AB, N);
    
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, AB, N, B, N, 1.0, first, N);
    
	cblas_dcopy(N * N, A, 1.0, second, 1.0);

	cblas_dtrmm(CblasRowMajor, CblasLeft,CblasUpper,CblasTrans,CblasNonUnit, N, N, 1.0, A, N, second, N);
    
	cblas_dcopy(N * N, first, 1.0, C, 1.0);

	cblas_daxpy(N * N, 1.0, second, 1.0, C, 1.0);

	free(AB);
	free(first);
	free(second);

	return C;



}
