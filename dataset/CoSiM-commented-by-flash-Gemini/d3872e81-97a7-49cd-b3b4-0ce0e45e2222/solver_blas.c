
#include "utils.h"
#include <cblas.h>
#include <stdio.h>
#include <string.h>



double *add_matrix(double *mat1, double *mat2, int N)
{
	register int i, j;
	double *mat_sum = malloc(N * N * sizeof(double));

	for (i = 0; i < N; i++)
        for (j = 0; j < N; j++) {
			register double sum = 0.0;
		    sum = mat1[i * N + j] + mat2[i * N + j];
			mat_sum[i * N + j] = sum;
		}

	return mat_sum;
}


double *multiply_simple_matrix(double *mat1, double *mat2, int N)
{
	double *mat_mult = malloc(N * N * sizeof(double));
	memcpy(mat_mult, mat2, N * N * sizeof(double));

	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasNoTrans,
		CblasNonUnit,
		N, N,
		1.0, mat1, N,
		mat_mult, N
	);

	return mat_mult;
}


double *multiply_matrix_transpose(double *mat1, double *mat2, int N)
{
	double *mat_mult = malloc(N * N * sizeof(double));
	cblas_dgemm(
		CblasRowMajor,
		CblasNoTrans,
		CblasTrans,
		N, N,
		N, 1,
		mat1, N,
		mat2, N,
		0, mat_mult, N);
	return mat_mult;
}


double *multiply_transpose_matrix(double *mat1, double *mat2, int N)
{
	double *mat_mult = malloc(N * N * sizeof(double));
	cblas_dgemm(
		CblasRowMajor,
		CblasTrans,
		CblasNoTrans,
		N, N,
		N, 1,
		mat1, N,
		mat2, N,
		0, mat_mult, N);

	return mat_mult;
}

double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	
	double *AB = NULL;
	double *ABBt = NULL;
	double *AtA = NULL;
	double *result = NULL;

	
	AB = multiply_simple_matrix(A, B, N);

	
	ABBt = multiply_matrix_transpose(AB, B, N);

	
	AtA = multiply_transpose_matrix(A, A, N);

	
	result = add_matrix(ABBt, AtA, N);

	free(AB);
	free(ABBt);
	free(AtA);

	return result;
}
