
#include <string.h>
#include <stdlib.h>

#include "utils.h"
#include "cblas.h"


double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	double *C = calloc(N * N, sizeof(double));
	double *AB = calloc(N * N, sizeof(double)); 
	double *D = calloc(N * N, sizeof(double)); 
	double *E = calloc(N * N, sizeof(double)); 

	int i;

	if(C == NULL || AB == NULL || D == NULL || E == NULL) {
		printf("Eroare la alocare\n");
		return NULL;
	}

	
	memcpy(AB, B, N * N * sizeof(double));

	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
		N, N, 1.0, A, N, AB, N);

	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N,
		1.0, AB, N, B, N, 0.0, D, N);

	
	memcpy(E, A, N * N * sizeof(double));

	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit,
		N, N, 1.0, A, N, E, N);

	
	for(i = 0; i < N * N; i++) {
		C[i] = D[i] + E[i];
	}

	free(AB);
	free(D);
	free(E);

	return C;
}
