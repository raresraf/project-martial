/**
 * @file solver_blas.c
 * @brief A matrix solver implementation using a BLAS library (CBLAS interface).
 *
 * This file provides an implementation of the `my_solver` function that uses
 * a BLAS library to compute the result of the expression: C = (A * B) * B^T + A^T * A.
 * It breaks the problem down into several distinct BLAS calls and a final
 * manual summation step.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "utils.h"
#include "cblas.h"

/**
 * @brief Allocates memory for all necessary matrices.
 *
 * A helper function to allocate heap memory for the final result matrix `C` and
 * all intermediate matrices used in the calculation. Exits the program on
 * allocation failure.
 *
 * @param N The dimension of the square matrices.
 * @param C   Pointer to the pointer for the final result matrix.
 * @param AB  Pointer to the pointer for the intermediate matrix (A * B).
 * @param ABBt Pointer to the pointer for the intermediate matrix ((A * B) * B^T).
 * @param AtA Pointer to the pointer for the intermediate matrix (A^T * A).
 */
void allocate_matrices(int N, double **C, double **AB, double **ABBt, double **AtA)
{
	*C = malloc(N * N * sizeof(**C));
	if (NULL == *C)
		exit(EXIT_FAILURE);

	*AB = malloc(N * N * sizeof(**AB));
	if (NULL == *AB)
		exit(EXIT_FAILURE);
   
	*ABBt = malloc(N * N * sizeof(**ABBt));
	if (NULL == *ABBt)
		exit(EXIT_FAILURE);

	*AtA = malloc(N * N * sizeof(**AtA));
	if (NULL == *AtA)
		exit(EXIT_FAILURE);
}

/**
 * @brief Solves a matrix equation using a sequence of BLAS functions and manual summation.
 *
 * This function computes the expression: C = (A * B) * B^T + A^T * A, where
 * A and B are N x N matrices. It uses BLAS for the heavy lifting of matrix
 * multiplication but performs the final matrix addition with a manual C loop.
 *
 * @param N The dimension of the square matrices.
 * @param A A pointer to the N x N input matrix A. Assumed to be upper triangular.
 * @param B A pointer to the N x N input matrix B.
 * @return A pointer to a newly allocated N x N matrix containing the result.
 *         The caller is responsible for freeing this memory.
 *
 * @note The implementation performs the following steps:
 * 1. Allocates memory for all matrices (`C`, `AB`, `ABBt`, `AtA`).
 * 2. Computes `AtA = A^T * A` using `cblas_dtrmm`.
 * 3. Computes `AB = A * B` using `cblas_dtrmm`.
 * 4. Computes `ABBt = (A * B) * B^T` using `cblas_dgemm`.
 * 5. Manually computes the final result `C = ABBt + AtA` using nested loops.
 * 6. Frees all intermediate matrices.
 */
double* my_solver(int N, double *A, double *B) 
{
	double *C;
	double *AB;
	double *ABBt;
	double *AtA;
	int i, j;

	allocate_matrices(N, &C, &AB, &ABBt, &AtA);

	
	memcpy(AtA, A, N * N * sizeof(*AtA));

	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasTrans,
		CblasNonUnit,
		N, N,
		1.0, A, N,
		AtA, N
	);

	
	memcpy(AB, B, N * N * sizeof(*AtA));

	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasNoTrans,
		CblasNonUnit,
		N, N,
		1.0, A, N,
		AB, N
	);

	
	cblas_dgemm(
		CblasRowMajor,
		CblasNoTrans,
		CblasTrans,
		N, N, N,
		1.0,
		AB,
		N,
		B,
		N,
		0.0,
		ABBt,
		N
	);

	for (i = 0; i < N; i++){
        	for(j = 0; j < N; j++) 
            		C[i * N + j] += ABBt[i * N + j] + AtA[i * N + j];
    	}
	free(AB);
	free(ABBt);
	free(AtA);
	return C;
}
