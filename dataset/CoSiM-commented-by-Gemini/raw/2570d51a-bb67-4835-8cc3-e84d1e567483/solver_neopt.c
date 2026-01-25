
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");

	
	long nr_elem = N*N;
	int i, j, k;
	double *C, *AB, *ABBt, *AtA, *At, *Bt;
	C = (double *)calloc(nr_elem, sizeof(double));
	AB = (double *)calloc(nr_elem, sizeof(double));
	ABBt = (double *)calloc(nr_elem, sizeof(double));
	AtA = (double *)calloc(nr_elem, sizeof(double));
	At = (double *)calloc(nr_elem, sizeof(double));
	Bt = (double *)calloc(nr_elem, sizeof(double));


	
	for (i = 0 ; i < N ; i ++)
	{
		for (j = 0 ; j < N ; j ++)
		{
			for (k = i ; k < N ; k ++)
			{
				AB[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	
	for (i = 0 ; i < N ; i ++)
	{
		for(j = 0 ; j < N ; j ++)
		{
			At[j * N + i] = A[i * N + j];
			Bt[j * N + i] = B[i * N + j];
		}
	}

	
	for (i = 0 ; i < N ; i ++)
	{
		for (j = 0 ; j < N ; j ++)
		{
			for (k = 0 ; k < N ; k ++)
			{
				ABBt[i * N + j] += AB[i * N + k] * Bt[k * N + j];
			}
		}
	}


	
	for (i = 0 ; i < N ; i ++)
	{
		for (j = 0 ; j < N ; j ++)
		{
			for (k = 0 ; k <= i ; k ++)
			{
				AtA[i * N + j] += At[i * N + k] * A[k * N + j];
			}
			C[i * N + j ] = ABBt[i * N + j] + AtA[i * N + j];
		}
	}


	
	free(AB);
	free(ABBt);
	free(AtA);
	free(At);
	free(Bt);


	return C;

}
