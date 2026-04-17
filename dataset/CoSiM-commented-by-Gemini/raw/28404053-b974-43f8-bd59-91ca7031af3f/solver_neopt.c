/* Module Level: Non-optimized matrix solver. @raw/28404053-b974-43f8-bd59-91ca7031af3f/solver_neopt.c */
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	
	int i = 0;
	int j = 0;
	int k = 0;

	double *fst = malloc(N * N * sizeof(double));
	
	/* Block Level: First basic multiplication */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			double sum = 0;
			for (k = 0; k < N; k++) {
				if (k >= i)
					sum += A[i * N + k] * B[k * N + j];
			}
			fst[i * N + j] = sum;
		}
	}

	double *snd = malloc(N * N * sizeof(double));

	/* Block Level: Second basic multiplication */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			double sum = 0;
			for (k = 0; k < N; k++) {
				sum += fst[i * N + k] * B[j * N + k];
			}
			snd[i * N + j] = sum;
		}
	}

	double *third = malloc(N * N * sizeof(double));
	
	/* Block Level: Final matrix construction */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			double sum = 0;
			for (k = 0; k < N; k++) {
				sum += A[k * N + i] * A[k * N + j];
			}
			third[i * N + j] = sum;
			third[i * N + j] += snd[i * N + j];
		}
	}

	free(fst);
	free(snd);
	
	return third;			
}