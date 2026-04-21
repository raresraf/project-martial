/**
 * @662cf250-db14-4761-b30d-678fa03cf1ab/solver_blas.c
 * @brief Matrix expression solver utilizing high-performance Level 3 BLAS.
 * Functional Utility: Solves Result = (A * B) * B^T + (A^T * A) where A is upper 
 * triangular, using optimized triangular matrix-matrix multiplication (TRMM).
 * Domain: HPC Numerical Optimization.
 */

#include "utils.h"
#include <string.h>
#include <stdlib.h>
#include "./cblas/CBLAS/include/cblas.h"
#include <math.h>
#include <stddef.h>

/**
 * @brief Computes the matrix expression via BLAS routine delegation.
 * Logic: Minimizes compute overhead by leveraging TRMM for triangular matrix factors.
 */
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

    /**
     * Block Logic: Compute Aux = A * B.
     * Optimization: Uses cblas_dtrmm to take advantage of A being upper triangular.
     */
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

    /**
     * Block Logic: Compute Res = Aux * B^T.
     * Functional Utility: General matrix multiplication with implicit transposition of B.
     */
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

    /**
     * Block Logic: Compute A_tA = A^T * A.
     * Optimization: In-place Gramian computation using TRMM with leading transposition.
     */
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

    /**
     * Block Logic: Final summation pass.
     */
    for(i = 0; i < N; i++)
    	for(j = 0; j < N; j++)
    			Res[i * N + j] += A_tA[i * N +j];

    free(A_tA);
    free(Aux);

    return Res;
}
