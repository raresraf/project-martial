
#include "utils.h"


double *allocate_matrix(int N) {
	double *res = calloc(N * N, sizeof(double));
	return res;
}

double* my_solver(int N, double *A, double* B) {
	double *C = allocate_matrix(N);
	double *AxB = allocate_matrix(N);
	double *AxBxBt = allocate_matrix(N);
	double *AtxA = allocate_matrix(N);
    int i, j, k;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            double sum = 0;
            for (k = i; k < N; k++) {
                sum += (A[i* N + k] * B[k * N + j]);
            }
            AxB[i * N + j] = sum;
        }
    } 

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            double sum = 0;
            for (k = 0; k < N; k++) {
                sum += (AxB[i * N + k] * B[j * N + k]);
            }
            AxBxBt[i * N + j] = sum;
        }
    } 

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            double sum = 0;
            for (k = 0; k <= i; k++) {
                sum += (A[k * N + i] * A[k * N + j]);
            }
            AtxA[i * N + j] = sum;
        }
    } 

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i * N + j] = AxBxBt[i * N + j] + AtxA[i * N + j];
		}
	}

	free(AxB);
	free(AxBxBt);
	free(AtxA);
	return C;
}

