
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	
	double *C;
	double *AtA, *BBt, *ABBt;
	int i, j, k;
	
	C = malloc(N * N * sizeof(*C));
	AtA = calloc(N * N, sizeof(*AtA));
	BBt = calloc(N * N, sizeof(*BBt));
	ABBt = calloc(N * N, sizeof(*ABBt));
	if (C == NULL || AtA == NULL || BBt == NULL || ABBt == NULL) {
		exit(EXIT_FAILURE);
	}
	
	
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			for (k = 0; k <= i; ++k) {
				AtA[i * N + j] += A[k *  N + i] * A[k * N + j];
			}
		}
	}

	
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			for (k = 0; k < N; ++k) {
				BBt[i * N + j] += B[i *  N + k] * B[j * N + k];
			}
		}
	}

	
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			for (k = i; k < N; ++k) {
				ABBt[i * N + j] += A[i *  N + k] * BBt[k * N + j];
			}
		}
	}

	
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			C[i * N + j] = ABBt[i * N + j] + AtA[i * N + j];
		}
	}

	free(ABBt);
	free(AtA);
	free(BBt);
	return C;
}
