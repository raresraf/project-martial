
#include "utils.h"

double* my_solver(int N, double *A, double *B) {
	
	double *C = (double *)calloc(N * N, sizeof(double));
	double *D = (double *)calloc(N * N, sizeof(double));

	int z = 0;

	
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = z; k < N; k++) {
				C[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
		z++;
	}

	
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < N; k++) {
				D[i * N + j] += C[i * N + k] * B[j * N + k];
			}
		}
	}
	
	
	for (int i = 0; i < N; i++) {
		for (int j = i; j < N; j++) {
			C[i * N + j] = 0;
			for (int k = 0; k <= i; k++) {
				C[i * N + j] += A[k * N + j] * A[k * N + i];   
			}
            if (i != j)
                C[j * N + i] = C[i * N + j];
		}
	}

	
	for (int i = 0; i < N * N; i++) {
		C[i] += D[i];
	}

	free(D);

	return C;
}