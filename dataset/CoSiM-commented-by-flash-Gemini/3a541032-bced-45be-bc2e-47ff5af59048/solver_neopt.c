
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	int i = 0;
	int j = 0;
	int k = 0;

	
	double *result_A = calloc(N * N, sizeof(double));
	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			result_A[i * N + j] = 0;
			for (k = 0; k < N; k++) {
				result_A[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	double *result_AB = calloc(N * N, sizeof(double));
	

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			result_AB[i * N + j] = 0;
			for (k = i; k < N; k++) {
				result_AB[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	double *result_ABBt = calloc(N * N, sizeof(double));

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			result_ABBt[i * N + j] = 0;
			for (k = 0; k < N; k++) {
				result_ABBt[i * N + j] += result_AB[i * N + k] * B[j * N + k];
			}
		}
	}

	double *C = calloc(N * N, sizeof(double));


	for (i = 0; i < N; i++) {
		for (j = 0 ; j < N; j++) {
			C[i * N + j] = result_ABBt[i * N + j] + result_A[i * N + j];
		}
	}
	free(result_A);
	free(result_AB);
	free(result_ABBt);
	return C;

}
