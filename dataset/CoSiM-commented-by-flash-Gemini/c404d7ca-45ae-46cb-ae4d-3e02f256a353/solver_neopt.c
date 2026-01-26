
#include "utils.h"


double *my_solver(int N, double *A, double *B)
{
	printf("NEOPT SOLVER\n");

	int i, j, k;
	double *C, *AB, *ABBt, *AtA;

	C = (double *) calloc(N * N, sizeof(double));

	if (C == NULL)
		exit(-1);

	AB = (double *) calloc(N * N, sizeof(double));

	if (AB == NULL)
		exit(-1);

	ABBt = (double *) calloc(N * N, sizeof(double));

	if (ABBt == NULL)
		exit(-1);

	AtA = (double *) calloc(N * N, sizeof(double));

	if (AtA == NULL)
		exit(-1);

	
	for (i = 0; i < N; ++i)
		for (j = 0; j < N; ++j)
			for (k = i; k < N; ++k)
				AB[i * N + j] += A[i * N + k] * B[k * N + j];

	
	for (i = 0; i < N; ++i)
		for (j = 0; j < N; ++j)
			for (k = 0; k < N; ++k)
				ABBt[i * N + j] += AB[i * N + k] * B[j * N + k];

	
	for (i = 0; i < N; ++i)
		for (j = 0; j < N; ++j)
			for (k = 0; k <= j; ++k)
				AtA[i * N + j] += A[k * N + i] * A[k * N + j];

	
	for (i = 0; i < N; ++i)
		for (j = 0; j < N; ++j)
			C[i * N + j] = ABBt[i * N + j] + AtA[i * N + j];

	
	free(AB);
	free(ABBt);
	free(AtA);

	return C;
}
