

#include <string.h>
#include "cblas.h"
#include "utils.h"


double* my_solver(int N, double *A, double *B) {
	double *AB;
	double *ABB_t; 
	double *C;

	AB = calloc(N * N , sizeof(double));
	if (AB == NULL) 
		exit(EXIT_FAILURE);
	ABB_t = calloc(N * N , sizeof(double));
	if (ABB_t == NULL) 
		exit(EXIT_FAILURE);
	C = calloc(N * N , sizeof(double));
	if (C == NULL)
		exit(EXIT_FAILURE);	

	memcpy(AB, B, N * N * sizeof(double));

		

	cblas_dtrmm(CblasRowMajor, CblasLeft,
				CblasUpper, CblasNoTrans,
				CblasNonUnit, N, N,
				1.0, A, N,
				AB, N);

	

	memcpy(ABB_t, B, N * N * sizeof(double));

	cblas_dgemm(CblasRowMajor, CblasNoTrans,
                 CblasTrans, N, N,
                 N, 1.0, AB,
                 N, ABB_t, N,
                 1.0, C, N);

	

	cblas_dgemm(CblasRowMajor, CblasTrans,
                 CblasNoTrans, N, N,
                 N, 1.0, A,
                 N, A, N,
                 1.0, C, N);
	

	free(AB);
	free(ABB_t);
	return C;
}
