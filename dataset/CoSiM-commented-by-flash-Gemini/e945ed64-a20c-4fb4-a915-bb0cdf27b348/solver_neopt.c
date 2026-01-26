
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	int i, j, k;

	double *C = calloc(N * N, sizeof(*C));
	if (C == NULL) {
		exit(EXIT_FAILURE);
	}

	double *A_tA = calloc(N * N, sizeof(*A_tA));
	if (A_tA == NULL) {
		exit(EXIT_FAILURE);
	}

	double *AB = calloc(N * N, sizeof(*AB));
	if (AB == NULL)
		exit(EXIT_FAILURE);

	double *ABB_t = calloc(N * N, sizeof(*ABB_t));
	if (ABB_t == NULL)
		exit(EXIT_FAILURE);

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = i; k < N; k++) {
				AB[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				ABB_t[i * N + j] += AB[i * N + k] * B[j * N + k];
			}
		}
	}

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k <= i; k++) {
				A_tA[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i * N + j] = ABB_t[i * N + j] + A_tA[i * N + j];
		}
	}

	free(A_tA);
	free(AB);
	free(ABB_t);

	return C;
}