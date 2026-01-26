
#include "utils.h"










double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");

	double *C, *D;
	int i, j, k;

	C = calloc(N * N, sizeof(double));
	if (C == NULL)
		exit(EXIT_FAILURE);

	D = calloc(N * N, sizeof(double));
	if (D == NULL)
		exit(EXIT_FAILURE);

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				
				
				D[i * N + j] += B[i * N + k] * B[j * N + k];
			}
		}
	}

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = i; k < N; k++) {
				C[i * N + j] += A[i * N + k] * D[k * N + j];
			}
		}
	}

	
	
	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			int min_index = i;
			if (min_index > j)
				min_index = j;
			for (k = 0; k <= min_index; k++) {
				C[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	free(D);
	return C;
}
