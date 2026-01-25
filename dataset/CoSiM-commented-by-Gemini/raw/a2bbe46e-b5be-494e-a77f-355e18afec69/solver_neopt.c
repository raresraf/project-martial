
#include "utils.h"


int min(int i, int j)
{
	if (i < j)
		return i;

	return j;
}


double *alloc_matrix(int N)
{
	return (double *)calloc(N * N, sizeof(double));
}


void free_matrix(double *A)
{
	free(A);
}


double *multiply(int N, double *A, double *B)
{
	int i = 0;
	int j = 0;
	int k = 0;
	double *res = alloc_matrix(N);

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			
			for (k = i; k < N; k++)
				res[i * N + j] += A[i * N + k] * B[k * N + j];
		}
	}

	return res;
}


double *multiply_with_transpose_right(int N, double *A, double *At)
{
	int i = 0;
	int j = 0;
	int k = 0;
	double *res = alloc_matrix(N);

	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			for (k = 0; k < N; k++)
				res[i * N + j] += A[i * N + k] * At[j * N + k];

	return res;
}


double *multiply_with_transpose_left(int N, double *A, double *At)
{
	int i = 0;
	int j = 0;
	int k = 0;
	double *res = alloc_matrix(N);

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++)
			
			for (k = 0; k < min(i, j) + 1; k++)
				res[i * N + j] += At[k * N + i] * A[k * N + j];
	}

	return res;
}


double *add(int N, double *A, double *B)
{
	int  i = 0;
	int j = 0;
	double *res = alloc_matrix(N);

	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			res[i * N + j] = A[i * N + j] + B[i * N + j];
	
	return res;
}


double* my_solver(int N, double *A, double *B) {
	double *AB = NULL;
	double *ABBt = NULL;
	double *AtA = NULL;
	double *C = NULL;

	
	AB = multiply(N, A, B);

	
	ABBt = multiply_with_transpose_right(N, AB, B);

	
	AtA = multiply_with_transpose_left(N, A, A);

	
	C = add(N, ABBt, AtA);

	free_matrix(AB);
	free_matrix(ABBt);
	free_matrix(AtA);

	return C;
}
