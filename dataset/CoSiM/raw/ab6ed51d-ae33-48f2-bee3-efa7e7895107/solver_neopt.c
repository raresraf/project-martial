
#include "utils.h"
#include <string.h>

double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");

	

	int i,j,k;
	
	double *C = malloc(sizeof(double) * N * N);
	if (!C) 
		return NULL;


	double *AB = malloc(sizeof(double) * N * N);
	if (!AB)
		return NULL;

	double *BBt = malloc(sizeof(double) * N * N);
	if (!BBt)
		return NULL;

	double *AtA = malloc(sizeof(double) * N * N);
	if (!AtA)
		return NULL;

	
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++) {
			AB[i * N + j] = 0.0;
			for (k = 0; k < N; k++)
				AB[i * N + j] += A[i * N + k] * B[k * N + j];
		}
	
	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			BBt[i * N + j] = 0.0;
			for (k = 0; k < N; k++) {
				BBt[i * N + j] += AB[i * N + k] * B[j * N + k];
			}
		}
	}

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			AtA[i * N + j] = 0.0;
			for (k = 0; k < N; k++) {
				AtA[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			C[i * N + j] = BBt[i * N + j] + AtA[i * N + j];

	free(BBt);
	free(AtA);
	free(AB);
	return C;
}
