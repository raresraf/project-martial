
#include "utils.h"
#include "cblas.h"


double* my_solver(int N, double *A, double *B) {
	double* resultat = calloc(N * N, sizeof(double));
	double* resultat2 = calloc(N * N, sizeof(double));
	double* resultat3 = calloc(N * N, sizeof(double));
	double* resultat4 = calloc(N * N, sizeof(double));

	for(int i = 0; i < N * N; i ++){
		resultat[i] = A[i];
	}

	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1, A, N, resultat, N);

	for(int i = 0; i < N * N; i ++){
		resultat2[i] = B[i];
	}

	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1, A, N, resultat2, N);

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1, resultat2, N, B, N, 1, resultat3, N);

	for(int i = 0; i < N * N; i++){
    	resultat4[i] = resultat3[i] + resultat[i];
    }
	
    free(resultat3);
    free(resultat2);
    free(resultat);
    
	return resultat4;
}
