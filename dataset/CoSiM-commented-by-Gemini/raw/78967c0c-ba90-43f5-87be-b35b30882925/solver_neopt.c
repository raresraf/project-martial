
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	
	double *At;
	double *Bt;
	double *C;
	
	double *Aux1;
	double *Aux2;
	
	int i, j, k;
	
	At = calloc(N * N, sizeof(double));
	Bt = calloc(N * N, sizeof(double));
	C = calloc(N * N, sizeof(double));
	
	Aux1 = calloc(N * N, sizeof(double));
	Aux2 = calloc(N * N, sizeof(double));
	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			At[j * N + i] = A[i * N + j];
		}
	}
	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			Bt[j * N + i] = B[i * N + j];
		}
	}
	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = i; k < N; k++) {
				Aux1[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}
	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				Aux2[i * N + j] += Aux1[i * N + k] * Bt[k * N + j];
			}
		}
	}

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			Aux1[i * N + j] = 0;

			for (k = 0; k <= i; k++) {
				Aux1[i * N + j] += At[i * N + k] * A[k * N + j];
			}
		}
	}
	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i * N + j] = Aux1[i * N + j] + Aux2[i * N + j];
		}
	}
	
	free(At);
	free(Bt);
	
	free(Aux1);
	free(Aux2);
	
	return C;
}
