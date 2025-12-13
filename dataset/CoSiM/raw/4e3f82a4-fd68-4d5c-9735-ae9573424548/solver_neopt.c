
#include "utils.h"



double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	
	

	int i, j, k = 0, idx = 0;
	double *A_transpose = (double *) calloc(N * N, sizeof(double));
	double *B_transpose = (double *) calloc(N * N, sizeof(double));
	double *rezPartial1 = (double *) calloc(N * N, sizeof(double));
	double *rezPartial2 = (double *) calloc(N * N, sizeof(double));
	double *rezPartial3 = (double *) calloc(N * N, sizeof(double));
	double *rez = (double *) calloc(N * N, sizeof(double));

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			idx = j * N + i;
			B_transpose[idx] = B[k];
			k++;
		}
	}
	
	k = 0;
	for (i = 0; i < N; i++) {
		k = i * N + i;
		for (j = i; j < N; j++) {
			idx = j * N + i;
			A_transpose[idx] = A[k];
			k++;
		}
	}

	
	
	for (i = 0; i < N; i++) {
		for (k = 0; k < N; k++) {
			double temp = A[i * N + k];
			double temp1 = A_transpose[i * N + k];
			for (j = 0; j < N; j++) {
				rezPartial1[i * N + j] += temp * B[k * N + j];
				rezPartial2[i * N + j] += temp1 * A[k * N + j];
			}
		}
	}

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				rezPartial3[i * N + j] += rezPartial1[i * N + k] * B_transpose[k * N + j];
			}
			rez[i * N + j] = rezPartial3[i * N + j] + rezPartial2[i * N + j];
		}
	}

	free(A_transpose);
	free(B_transpose);
	free(rezPartial1);
	free(rezPartial2);
	free(rezPartial3);
	return rez;
}