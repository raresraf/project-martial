
/* Module Level: Non-optimized matrix solver implementation. @raw/70258bcd-8e0f-49f7-af71-997809f60c20/solver_neopt.c */
#include "utils.h"


 
double* my_solver(int N, double *A, double* B) {
	int i, j, k;
	printf("NEOPT SOLVER\n");

	double *AB = (double*) calloc(N * N, sizeof(double));
	if (AB == NULL) {
		printf("Error calloc\n");
		return NULL;
	}
	double *ABBt = (double*) calloc(N * N, sizeof(double));
	if (ABBt == NULL) {
		printf("Error calloc\n");
		return NULL;
	}

	double *AtA = (double*) calloc(N * N, sizeof(double));
	if (AtA == NULL) {
		printf("Error calloc\n");
		return NULL;
	}

	double *C = (double*) calloc(N * N, sizeof(double));
	if (C == NULL) {
		printf("Error calloc\n");
		return NULL;
	}

	
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			for (k = i; k < N; ++k) {
				AB[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			for (k = 0; k <= i; ++k) {
				AtA[j * N + i] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			double sum = 0;
			for (k = 0; k < N; ++k) {
				sum += AB[i * N + k] * B[j * N + k];
			}
			C[i * N + j] = sum + AtA[i * N + j];
		}
	}

	free(AB);
	free(ABBt);
	free(AtA);

	return C;
}
