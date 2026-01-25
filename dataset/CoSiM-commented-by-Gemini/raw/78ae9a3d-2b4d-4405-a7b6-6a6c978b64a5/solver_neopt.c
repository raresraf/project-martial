
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	int i, j, k;
	double *aux1 = (double*) calloc(N * N, sizeof(double));
	double *aux2 = (double*) calloc(N * N, sizeof(double));

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
				for (k = 0; k < N; k++) {
					if (i <= k) {
						aux1[N * i + j] += A[N * i + k] * B[N * k + j];
					} 
					if (k <= i && k <= j) {  
						aux2[N * i + j] += A[N * k + i] * A[N * k + j];
					}
				}
		}
	}

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				aux2[N * i + j] += aux1[N * i + k] * B[N * j + k];
			}
		}
	}

	free(aux1);
	return aux2;
}
