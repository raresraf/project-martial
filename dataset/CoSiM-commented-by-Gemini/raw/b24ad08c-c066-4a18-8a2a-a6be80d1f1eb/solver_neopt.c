
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	
	size_t i;
	size_t j;
	size_t k;

	
	double *C = calloc(N * N, sizeof(double));

	if (C == NULL) {
		perror("Calloc C");
		exit(EXIT_FAILURE);
	}

	
	double *AB = calloc(N * N, sizeof(double));

	if (AB == NULL) {
		perror("Calloc AB");
		exit(EXIT_FAILURE);
	}

	
	double *P1 = calloc(N * N, sizeof(double));

	if (P1 == NULL) {
		perror("Calloc P1");
		exit(EXIT_FAILURE);
	}

	
	double *P2 = calloc(N * N, sizeof(double));

	if (P2 == NULL) {
		perror("Calloc P2");
		exit(EXIT_FAILURE);
	}

	
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			double sum = 0.0;

			for (k = i; k < N; ++k) {
				sum += A[i * N + k] * B[k * N + j];
			}

			AB[i * N + j] = sum;
		}
	}

	
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			double sum = 0.0;

			for (k = 0; k < N; ++k) {
				sum += AB[i * N + k] * B[j * N + k];
			}

			P1[i * N + j] = sum;
		}
	}

	
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			double sum = 0.0;

			for (k = 0; k <= i && k <= j; ++k) {
				sum += A[k * N + i] * A[k * N + j];
			}

			P2[i * N + j] = sum;
		}
	}

	
	for (i = 0; i < N * N; ++i) {
		C[i] = P1[i] + P2[i];
	}

	free(AB);
	free(P1);
	free(P2);

	return C;
}
