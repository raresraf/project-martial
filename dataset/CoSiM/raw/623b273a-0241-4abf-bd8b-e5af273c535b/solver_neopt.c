
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	double* result;
	double* AAtranspose;
	double* BBtranspose;
	double* ABBt;
	int i, j, k;

	result = calloc(N * N, sizeof(*result));
	AAtranspose = calloc(N * N, sizeof(*AAtranspose));
	BBtranspose = calloc(N * N, sizeof(*BBtranspose));
	ABBt = calloc(N * N, sizeof(*ABBt));

	
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			
			for(k = 0; k <= j && k <= i; k++) {
				AAtranspose[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	
	
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			for(k = 0; k < N; k++) {
				BBtranspose[i * N + j] += B[i * N + k] * B[j * N + k];
			}
		}
	}

	
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			
			for(k = i; k < N; k++) {
				ABBt[i * N + j] += A[i * N + k] * BBtranspose[k * N + j];
			}
		}
	}

	
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++)
			result[i * N + j] = ABBt[i * N + j] + AAtranspose[i * N + j];
		
	}

	free(AAtranspose);
	free(BBtranspose);
	free(ABBt);

	return result;
}
