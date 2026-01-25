
#include "utils.h"
#define min(x, y) (((x) < (y)) ? (x) : (y))


double* my_solver(int N, double *A, double* B) {

	double * AB, *ABBt, * AtA, * result;
	int i, j, k;
	AB = calloc(N * N, sizeof(double));
	ABBt = calloc(N * N, sizeof(double));
	AtA = calloc(N * N, sizeof(double));
	result = calloc(N * N, sizeof(double));

	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			for(k = i; k < N; k++) {
				AB[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			for(k = 0; k < N; k++) {
				ABBt[i * N + j] += AB[i * N + k] * B[j * N + k];
			}
		}
	}

	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			for(k = 0; k <= min(i, j); k++) {
				AtA[i * N +j] += A[k * N + i] * A[k * N + j];
			}
		}
	}
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			result[i * N + j] = ABBt[i * N + j] + AtA[i * N + j];
		}
	}

	free(AB);
	free(ABBt);
	free(AtA);

	return result;
}

