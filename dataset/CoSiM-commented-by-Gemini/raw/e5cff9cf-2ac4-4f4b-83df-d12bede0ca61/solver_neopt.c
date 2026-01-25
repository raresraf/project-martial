
#include "utils.h"
#define min(a, b) (((a) < (b)) ? (a) : (b))

double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	double *C = malloc(N * N * sizeof(double));
	if (!C)
		return NULL;

	double *D = malloc(N * N * sizeof(double));
	if (!D)
		return NULL;

	int i, j, k;
	double sum;

	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			sum = 0;
			for (k = i; k < N; ++k) {
				sum += A[i * N + k] * B[k * N + j];
			}
			C[i * N + j] = sum;
		}
	}
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			sum = 0;
			for (k = 0; k < N; ++k) {
				sum += C[i * N + k] * B[j * N + k];
			}
			D[i * N + j] = sum;
		}
	}
	
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			sum = 0;
			for (k = 0; k <= min(i, j); ++k) {
				sum += A[k * N + i] * A[k * N + j];
			}
			D[i * N + j] += sum;
		}
	}
	free(C);
	return D;
}
