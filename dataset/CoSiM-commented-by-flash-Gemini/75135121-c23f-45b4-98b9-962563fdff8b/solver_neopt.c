
#include "utils.h"



double* compute_transpose(int N, double* M)
{
	double* res = (double*)malloc(N * N * sizeof(double));
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			res[j * N + i] = M[i * N + j];
		}
	}
	return res;
}

double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	double* res_AxB = (double*)malloc(N * N * sizeof(double));
	double* res_ABBt = (double*)malloc(N * N * sizeof(double));
	double* res_AtA = (double*)malloc(N * N * sizeof(double));
	double* res = (double*)malloc(N * N * sizeof(double));
	double* B_t = compute_transpose(N, B);
	double* A_t = compute_transpose(N, A);

	
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			res_AxB[i * N + j] = 0;
			for (int k = i; k < N; k++) {
				res_AxB[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	
	int P = 0;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			res_ABBt[i * N + j] = 0;
			for (int k = 0; k < N; k++) {
				res_ABBt[i * N + j] += res_AxB[i * N + k] * B_t[k * N + j];
			}
		}
	}

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			res_AtA[i * N + j] = 0;
			if (i < j) {
				P = i;
			}
			else {
				P = j;
			}
			for (int k = 0; k <= P; k++) {
				res_AtA[i * N + j] += A_t[i * N + k] * A[k * N + j];
			}
		}
	}
	
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			res[i * N + j] = 0;
			res[i * N + j] += res_ABBt[i * N + j] + res_AtA[i * N + j];
		}
	}
	free(A_t);
	free(B_t);
	free(res_AtA);
	free(res_ABBt);
	free(res_AxB);
	return res;
}
