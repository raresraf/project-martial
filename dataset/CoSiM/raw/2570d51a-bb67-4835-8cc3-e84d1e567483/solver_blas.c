
#include "utils.h"
#include <string.h>
#include "cblas.h"

double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");

	
	long nr_elem = N*N;
	double *C;
	C = (double *)calloc(nr_elem, sizeof(double));

	

	
	
	memcpy(C, B, nr_elem * sizeof(double));
	cblas_dtrmm(CblasRowMajor,
				CblasLeft,
				CblasUpper,
				CblasNoTrans,
				CblasNonUnit,
				N, N, 1.0,
				A, N, 
				C, N); 

	
	cblas_dtrmm(CblasRowMajor,
				CblasLeft,
				CblasUpper,
				CblasTrans,
				CblasNonUnit,
				N, N, 1.0, 
				A, N, 
				A, N);


	
	cblas_dgemm(CblasRowMajor,
				CblasNoTrans,
				CblasTrans,
				N, N, N, 1.0, 
				C, N,
				B, N, 
				1.0, A, N);

	
	memcpy(C, A, nr_elem * sizeof(double));

	return C;

}
