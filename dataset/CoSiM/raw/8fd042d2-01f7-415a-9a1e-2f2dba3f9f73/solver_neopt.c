
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	double *temp2 = (double *)calloc(N * N, sizeof(double));
	double *temp3 = (double *)calloc(N * N, sizeof(double));
	double *temp5 = (double *)calloc(N * N, sizeof(double));
	
	double *res = (double *)calloc(N * N, sizeof(double));
	int i, j, k;
	
	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for(k = 0; k <= i; k++) {
				temp2[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}
	
	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = i; k < N; k++) {
				temp3[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}
	
	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for(k = 0; k < N; k++)
				temp5[i * N + j] += temp3[i * N + k] * B[j * N + k];
		}
	}
	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {	
			res[i * N + j] = temp5[i * N + j] + temp2[i * N + j];
		}
	}
	
	free(temp2);
	free(temp3);
	free(temp5);
	
	return res;
}
