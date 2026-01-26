
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	double *C, *A_tA, *AB;
	int i, j, k, limit;
	
	C = calloc(N * N, sizeof(double));
	if (C == NULL)
		return NULL;
	
	A_tA = calloc(N * N, sizeof(double));
	if (A_tA == NULL)
		return NULL;
	
	AB = calloc(N * N, sizeof(double));
	if (AB == NULL)
		return NULL;
	
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++) {
			if (i <= j)
				limit = i;
			else limit = j;
			for (k = 0; k <= limit; k++) 
				A_tA[i * N + j] += A[k * N + i] * A[k * N  + j];
		}
	
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++) {
			for (k = i; k < N; k++)
				AB[i * N + j] += A[i * N + k] * B[k * N  + j];
		}
	
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			for (k = 0; k < N; k++)
				C[i * N + j] += AB[i * N + k] * B[j * N + k];
	
	for (i = 0; i < N; i++)
		for (j = 0;  j < N; j++)
			C[i * N + j] += A_tA[i * N + j];
	free(A_tA);
	free(AB);
	return C;
}
