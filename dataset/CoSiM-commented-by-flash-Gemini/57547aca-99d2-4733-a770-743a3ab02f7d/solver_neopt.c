
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	int i = 0;
	int j = 0;
	int k = 0;

	
	double *At = malloc(N * N * sizeof(double));
	double *Bt = malloc(N * N * sizeof(double));

	for ( i = 0; i < N; i++) {
		for ( j = 0; j < N; j++) {
			int index1 = i * N + j;
			int index2 = j * N + i;

			At[index2] = A[index1];
			Bt[index2] = B[index1];
		}
	}

	double *AB = malloc(N * N * sizeof(double));
	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = i; k < N; k++) {
				AB[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	
	double *AtA = malloc(N * N * sizeof(double));
	double *ABBt = malloc(N * N * sizeof(double));


	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				AtA[i * N + j] += At[i * N + k] * A[k * N + j];
				ABBt[i * N + j] += AB[i * N + k] * Bt[k * N + j];
			}
		}
	}

	double *res = malloc(N * N * sizeof(double));

	for (i = 0; i < N; i++) {
		for (j = 0 ; j < N; j++) {
			res[i * N + j] = ABBt[i * N + j] + AtA[i * N + j];
		}
	}

	free(At);
	free(Bt);
	free(AtA);
	free(ABBt);
	printf("NEOPT SOLVER\n");
	return res;
}
