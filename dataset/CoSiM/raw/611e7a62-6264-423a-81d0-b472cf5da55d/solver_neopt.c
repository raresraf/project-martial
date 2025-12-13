
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	double *C;
	double *D;
	double *E;
	int i, j, k;
	
	C = calloc(N*N, sizeof(double));
	D = calloc(N*N, sizeof(double));
	E = calloc(N*N, sizeof(double));

	for(i = 0; i < N; i++){
		for (j = 0; j < N; j++) {
			int var = 0;
			if (i >= j) {
				var = i;
			}
			for (k = var; k < N; k++) {
				C[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			for(k = 0; k < N; k++) {
				D[i * N + j] +=  C[i * N + k] * B[k + j * N];
			}
		}
	}

	for(i = 0; i < N; i++){
		for (j = 0; j < N; j++) {
			for (k = 0; k <= j; k++) {
				E[i * N + j] += A[k * N + i] * A[k * N + j];
			}
			E[i * N + j] += D[i * N + j];
		}
	}

	free(C);
	free(D);

 	return E;
}
