
#include "utils.h"



double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	double* A_T = (double *)calloc(N * N , sizeof(double));
	double* B_T = (double *)calloc(N * N , sizeof(double));
	double* result_1 = (double *)calloc(N * N , sizeof(double));
	double* result_2 = (double *)calloc(N * N , sizeof(double));
	double* result_3 = (double *)calloc(N * N , sizeof(double));
	double* C = (double *)calloc(N * N , sizeof(double));
	int i ,j,k;

	
	for(i = 0; i < N; i++)
	{
		for(j = 0; j < N; j++)
		{
			A_T[N * i + j] = A[N * j + i];
			B_T[N * i + j] = B[N * j + i];
		}
	}
	
	
	for(i = 0; i < N; i++)
	{
		for(j = 0; j < N; j++)
		{
			for(k = i; k < N; k++)
			{
				result_1[N * i + j] += A[N * i + k] * B[N * k + j];
			}
		}
	}
	
	
	for(i = 0; i < N; i++)
	{
		for(j = 0; j < N; j++)
		{
			for(k = 0; k < N; k++)
			{
				result_2[N * i + j] += result_1[N * i + k] * B_T[N * k + j];
			}
		}
	}
	
	
	for(i = 0; i < N; i++)
	{
		for(j = 0; j < N; j++)
		{
			for(k = 0; k <= (i < j ? i : j); k++)
			{
				result_3[N * i + j] += A_T[N * i + k] * A[N * k + j];
			}
		}
	}


	
	for(i = 0 ; i < N; i++)
	{
		for(j = 0; j < N; j++)
		{
			C[N * i + j] = result_2[N * i + j] + result_3[N * i + j];
		}
	}	


	
	free(A_T);
	free(B_T);
	free(result_1);
	free(result_2);
	free(result_3);
	return C;
}
