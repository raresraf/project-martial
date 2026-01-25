
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	
	printf("NEOPT SOLVER\n");
	double *AtA = calloc(N * N, sizeof(double));
	double *AB = calloc(N * N, sizeof(double));
	double *ABBt = calloc(N * N, sizeof(double));
	double *res = calloc(N * N, sizeof(double));

	
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			for (int k = 0; k < N; ++k) {
				if (k > j) {
					break;
				}
				AtA[i*N + j] += A[k*N + i] * A[k*N + j];
			}
		}
	}

	
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			for (int k = 0; k < N; ++k) {
				if (k < i) {
					continue;
				}
				AB[i*N + j] += A[i*N + k] * B[k*N + j];
			}
		}
	}

	
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			for (int k = 0; k < N; ++k) {
				ABBt[i*N + j] += AB[i*N + k] * B[j*N + k];
			}
			res[i*N + j] = ABBt[i*N + j] + AtA[i*N + j];
		}
	}

	free(AtA);
	free(AB);
	free(ABBt);

	return res;
}
