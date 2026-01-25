

#include "utils.h"
#include <string.h>
#include <stdlib.h>
#include "./cblas/CBLAS/include/cblas.h"
#include <math.h>
#include <stddef.h>

double *my_solver(int N, double *A, double *B) {

    double *Aux;
    double *Res;
    double *A_tA;
    int i, j;

    Aux = malloc(N * N * sizeof(*Aux));
	if(NULL == Aux)
		exit(EXIT_FAILURE);

	Res = malloc(N * N * sizeof(*Res));	
	if(NULL == Res)
		exit(EXIT_FAILURE);

	A_tA = malloc(N * N * sizeof(*A_tA));
	if(NULL == A_tA)
		exit(EXIT_FAILURE);

    

    memcpy(Aux, B, N * N * sizeof(*Aux));

    cblas_dtrmm(
    	CblasRowMajor,
    	CblasLeft,
    	CblasUpper,
    	CblasNoTrans,
    	CblasNonUnit,
    	N, N,
    	1.0, A, N,
    	Aux, N
    );

    

    memcpy(Res, B, N * N * sizeof(*Aux));

    cblas_dgemm(
    	CblasRowMajor,
    	CblasNoTrans,
    	CblasTrans,
    	N, N, N,
    	1.0, Aux, N,
    	B, N,
    	0.0, Res, N
    );

    

    memcpy(A_tA, A, N * N * sizeof(*Res));

    cblas_dtrmm(
    	CblasRowMajor,
    	CblasLeft,
    	CblasUpper,
    	CblasTrans,
    	CblasNonUnit,
    	N, N,
    	1.0, A, N,
    	A_tA, N
    );

    

    for(i = 0; i < N; i++)
    	for(j = 0; j < N; j++)
    			Res[i * N + j] += A_tA[i * N +j];

    free(A_tA);
    free(Aux);

    return Res;
}