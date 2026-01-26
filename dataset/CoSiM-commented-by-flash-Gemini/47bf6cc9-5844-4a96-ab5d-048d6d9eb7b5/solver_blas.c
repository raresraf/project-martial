

#include <string.h>
#include <stdlib.h>
#include "cblas.h"
#include "utils.h"


double* my_solver(int N, double *A, double *B) {

	double *AB, *C, *AA;
	AB = calloc(N * N,  sizeof(double));
	C = calloc(N * N, sizeof(double));
	AA = calloc(N * N, sizeof(double));
	if (AB == NULL || C == NULL) {
        perror("malloc failed\n");
        exit(EXIT_FAILURE);
    }

	
	
	
	double alpha = 1.0, beta = 0.0;
	int i = 0, j = 0;

	memcpy(AB, B, N * N * sizeof(double));
	memcpy(AA, A, N * N * sizeof(double));
	
	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				N, N, N, alpha, A, N, B, N, beta, AB, N);

	
	cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
				N, N, N, alpha, AB, N, B, N, beta, C, N);
	
	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit,
				N, N, alpha, A, N, AA, N); 

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i * N + j] += AA[i * N + j];
		}
	}
	

	free(AA);
	free(AB);
	return C;
}
