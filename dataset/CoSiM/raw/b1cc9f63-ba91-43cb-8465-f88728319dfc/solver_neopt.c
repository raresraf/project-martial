
#include "utils.h"

double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	
	int i, j, k;
	double *AB, *ABBt;
	

	AB = malloc(sizeof(double) * N * N);
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++)
			for (k = i; k < N; k++)
				 AB[N * i + j] += A[N * i + k] * B[N * k + j];
	}

	ABBt = malloc(sizeof(double) * N * N);
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			for (k = 0; k < N; k++)
				ABBt[N * i + j] += AB[N * i + k] * B[k + N * j];

	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			for (k = 0; k <= i; k++)
				ABBt[N * i + j] += A[i + N * k] * A[k * N + j];


	free(AB);

	return ABBt;
}
