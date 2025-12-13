
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	double *C = calloc(N * N, sizeof(double));
	double *D = calloc(N * N, sizeof(double));
	double *E = calloc(N * N, sizeof(double));

	int i, j, k;

	if(C == NULL || D == NULL || E == NULL) {
		printf("Eroare la alocare");
		return NULL;
	}

	
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			double sum = 0;
			for(k = 0; k < N; k++) {
				if(i <= k) {
					sum += A[i * N + k] * B[k * N + j];
				}
			}
			D[i * N + j] = sum;
		}
	}

	
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			double sum = 0;
			for(k = 0; k < N; k++) {
				sum += D[i * N + k] * B[j * N + k];
			}
			E[i * N + j] = sum;
		}
	}

	
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			double sum = 0;
			for(k = 0; k < N; k++) {
				if(k >= i || k <= j) {
					sum += A[k * N + i] * A[k * N + j];
				}
			}
			C[i * N + j] = E[i * N + j] + sum;
		}
	}

	free(D);
	free(E);

	return C;
}

