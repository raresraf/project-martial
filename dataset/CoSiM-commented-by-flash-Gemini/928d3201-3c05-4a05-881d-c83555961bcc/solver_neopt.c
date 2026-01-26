
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");

	int i, j, k;

	double *BBt = (double*) calloc(N * N, sizeof(double));

	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			for (k = 0; k < N; ++k) {
				BBt[i * N + j] += B[i * N + k] * B[j * N + k];
			}
		}
	}

	double *ABBt = (double*) calloc(N * N, sizeof(double));

	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			for (k = i; k < N; ++k) {
				ABBt[i * N + j] += A[i * N + k] * BBt[k * N + j];
			}
		}
	}

	free(BBt);

	double *AAt = (double*) calloc(N * N, sizeof(double));

	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			for (k = 0; k <= ((i < j) ? i : j); ++k) {
				AAt[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	double *res = (double*) calloc(N * N, sizeof(double));

	for (i = 0; i < N * N; ++i) {
		res[i] = ABBt[i] + AAt[i];
	}

	free(ABBt);
	free(AAt);

	return res;
}
