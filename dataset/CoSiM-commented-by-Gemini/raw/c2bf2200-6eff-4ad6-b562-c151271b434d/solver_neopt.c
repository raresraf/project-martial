
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	double *AB = malloc(N * N * sizeof(double));
	double *ABBt = malloc(N * N * sizeof(double));
	double *AtA = malloc(N * N * sizeof(double));
	double *R = malloc(N * N * sizeof(double));
	int i, j, k;

	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			AB[i * N + j] = 0.0;
			for (k = i; k < N; ++k) {
				AB[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			ABBt[i * N + j] = 0.0;
			for (k = 0; k < N; ++k) {
				ABBt[i * N + j] += AB[i * N + k] * B[j * N + k];
			}
		}
	}

	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			AtA[i * N + j] = 0.0;
			for (k = 0; k < N; ++k) {
				if (i < k || j < k)
					break;
				AtA[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			R[i * N + j] = ABBt[i * N + j] + AtA[i * N + j];
		}
	}

	free(AB);
	free(ABBt);
	free(AtA);

	return R;
}
