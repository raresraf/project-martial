
#include "utils.h"
#include "string.h"

#include "cblas.h"


double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	double *C, *D;


	C = calloc(N * N, sizeof(double));
	if (C == NULL)
		exit(EXIT_FAILURE);

	D = calloc(N * N, sizeof(double));
	if (D == NULL)
		exit(EXIT_FAILURE);

	
	memcpy(D, B, N * N * sizeof(double));
	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasNoTrans,
		CblasNonUnit,
		N, N,
		1.0, A, N,
		D, N 
	);

	
	
	
	memcpy(C, A, N * N * sizeof(double));
	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasTrans, 
		CblasNonUnit,
		N, N,
		1.0, A, N,
		C, N 
	);


	
	cblas_dgemm(
		CblasRowMajor,
		CblasNoTrans, 
		CblasTrans,  
		N, N, N,
		1.0, D, N,
		B, N,
		1.0, C, N 
	);

	free(D);

	return C;
}
