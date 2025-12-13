
#include "utils.h"


void transpose_matrix(int N,double *initial_matrix, double *transpose)
{
	int i,j;
	for ( i = 0; i < N; i++)
  		for ( j = 0; j < N; j++) {
    		transpose[j * N + i] = initial_matrix[i * N + j];
  }
}

double* multiply_matrix(int N, double *A, double* B)
{
	int i,j,k;
	double*  result = calloc(N*N ,sizeof(result));
	for (i = 0 ; i < N ; i++)
		for (j = 0 ; j < N ; j++)
			for(k = 0 ; k < N ; k ++)
				 result[i * N + j] += A[i * N + k] * B[k * N + j];
	
	return result;
}

double* multiply_matrix_inferior(int N, double *A, double* B)
{
	int i,j,k;
	double*  result = calloc(N*N , sizeof(result));
	for (i = 0 ; i < N ; i++)
		for (j = 0 ; j < N ; j++)
			for(k = 0 ; k <= i ; k++)
				 result[i * N + j] += A[i * N + k] * B[k * N + j];
	
	return result;
}

double* multiply_matrix_superior(int N, double *A, double* B)
{
	int i,j,k;
	double*  result = calloc(N*N , sizeof(result));
	for (i = 0 ; i < N ; i++)
		for (j = 0 ; j < N ; j++)
			for(k = i ; k < N ; k++)
				 result[i * N + j] += A[i * N + k] * B[k * N + j];
	
	return result;
}

double* add_matrix(int N, double *A, double* B)
{
	int i,j;
	double*  result = calloc(N*N , sizeof(result));
	for (i = 0; i < N; i++)
    	for (j = 0; j < N; j++) 
      		result[i * N + j] = A[i * N + j] +  B[i * N + j];

	return result;
    
}

double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	double * result;
	double * resultAXB;
	double * resultXTranspose;
	double * A_transposeXA;
	double * A_transpose = (double *) calloc(N*N , sizeof(double));
	double * B_transpose = (double *) calloc(N*N , sizeof(double));

	transpose_matrix(N,A,A_transpose);
	transpose_matrix(N,B,B_transpose);

	
	resultAXB = multiply_matrix_superior(N,A,B);
	resultXTranspose = multiply_matrix(N,resultAXB,B_transpose);
	
	A_transposeXA = multiply_matrix_inferior(N,A_transpose,A);


	result = add_matrix(N,resultXTranspose,A_transposeXA);

	free(A_transposeXA);
	free(B_transpose);
	free(A_transpose);
	free(resultXTranspose);
	free(resultAXB);
	
	return result;
}
