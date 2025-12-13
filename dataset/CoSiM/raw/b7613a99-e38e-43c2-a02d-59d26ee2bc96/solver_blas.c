
#include <stdlib.h>
#include <string.h>
#include "utils.h"
#include "cblas.h"


double* my_solver(int N, double *A, double *B) {

	double *C;
	double *AB;

	
	C = malloc(N * N * sizeof(*C));
	AB = malloc(N * N * sizeof(*AB));

	
	memcpy(C, A, N * N * sizeof(*C));
	memcpy(AB, B, N * N * sizeof(*AB));

	
	cblas_dtrmm( CblasRowMajor,
				CblasLeft,
				CblasUpper,
				CblasTrans,
				CblasNonUnit,
				N, N, 1.0,
				A, N,
				C, N);

	
	cblas_dtrmm( CblasRowMajor,
				CblasLeft,
				CblasUpper,
				CblasNoTrans,
				CblasNonUnit,
				N, N, 1.0,
				A, N,
				AB, N);

	
	cblas_dgemm( CblasRowMajor, 
				CblasNoTrans,
				CblasTrans,
				N, N, N, 1.0,
				AB, N,
				B, N,
				1.0, C, N);


	free(AB);

	printf("BLAS SOLVER\n");

	return C;
}
