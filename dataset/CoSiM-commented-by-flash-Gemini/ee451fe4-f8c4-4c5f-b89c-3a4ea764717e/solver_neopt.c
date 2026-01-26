
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");

	int i, j, k;

	double *At = (double *) calloc(N * N, sizeof(double));
	if (At == NULL) 
		return NULL;

	double *Bt = (double *) calloc(N * N, sizeof(double));
	if (Bt == NULL) 
		return NULL;

	double *AxB = (double *) calloc(N * N, sizeof(double));
	if (AxB == NULL) 
		return NULL;

	double *AxBxBt = (double *) calloc(N * N, sizeof(double));
	if (AxBxBt == NULL) 
		return NULL;

	double *result = (double *) calloc(N * N, sizeof(double));
	if (result == NULL) 
		return NULL;

	
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			At[j * N + i] = A[i * N + j];
			Bt[j * N + i] = B[i * N + j];
		}
	}

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = i; k < N; k++) {				
				AxB[i * N + j] += A[i * N + k] * B[k * N + j];
			}

		}
	}

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {				
				AxBxBt[i * N + j] += AxB[i * N + k] * Bt[k * N + j];
			}

		}
	}

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < i + 1; k++) {	
				result[i * N + j] += At[i * N + k] * A[k * N + j];
			}

		}
	}

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {	
			result[i * N + j] += AxBxBt[i * N + j];
		}
	}
	
	free(At);
	free(Bt);
	free(AxB);
	free(AxBxBt);
	return result;
}
