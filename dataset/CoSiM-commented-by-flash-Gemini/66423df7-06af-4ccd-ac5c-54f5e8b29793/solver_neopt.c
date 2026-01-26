
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	
	printf("NEOPT SOLVER\n");
	int i, j, k;
	double *AB = (double *)calloc(N * N, sizeof(double));
	double *Bt = (double *)calloc(N * N, sizeof(double));
	double *ABBt = (double *)calloc(N * N, sizeof(double));
	double *At = (double *)calloc(N * N, sizeof(double));
	double *AtA = (double *)calloc(N * N, sizeof(double));
	double *result = (double *)calloc(N * N, sizeof(double));

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = i; k < N; k++) {
				AB[N * i + j] +=  A[N * i + k] * B[N * k + j];
			}
		}
	}
	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			Bt[j * N + i] = B[i * N + j];
		}
	}
	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				ABBt[N * i + j] +=  AB[N * i + k] * Bt[N * k + j];
			}
		}
	}
	
	for (i = 0; i < N; ++i) {
		for (j = i; j < N; ++j) {
			At[j * N + i] = A[i * N + j];
		}
	}
	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k <= i; k++) {
				AtA[N * i + j] +=  At[N * i + k] * A[N * k + j];
			}
		}
	}
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			result[i * N + j] = ABBt[i * N + j] + AtA[i * N + j];
		}
	}
	
	free(AB);
	free(ABBt);
	free(AtA);
	free(Bt);
	free(At);
	return result;
}
