
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	int i, j, k;

	double *C = (double *)calloc(N * N, sizeof(double));
	if (C == NULL) return NULL;

	double *result1 = (double *)calloc(N * N, sizeof(double));
	if (result1 == NULL) return NULL;

	double *result2 = (double *)calloc(N * N, sizeof(double));
	if (result2 == NULL) return NULL;

	double *At = (double *)calloc(N * N, sizeof(double));
	if (At == NULL) return NULL;

	double *Bt = (double *)calloc(N * N, sizeof(double));
	if (Bt == NULL) return NULL;

	for (i = 0; i < N; i++) {
		register double *At_1 = &At[i * N];
		register double *Bt_1 = &Bt[i * N];
		for (j = 0; j < N; j++) {
			At_1[j] = A[j * N + i];
			Bt_1[j] = B[j * N + i];
		}
	}

	
	for (i = 0; i < N; i++) {
		register double *A_1 = &A[i * N];
		register double *result1_1 = &result1[i * N];
		for (k = 0; k < N; k++) {
			if (i > k) {
				continue;
			}
			register double *B_1 = &B[k * N];
			register double A_2 = A_1[k];
			for (j = 0; j < N; j++) {
				result1_1[j] += A_2 * B_1[j];
			}
		}
	}

	
	
	
	
	
	
	
	
	
	
	

	for (i = 0; i < N; i++) {
		register double *result1_1 = &result1[i * N];
		register double *result2_1 = &result2[i * N];
		for (k = 0; k < N; k++) {
			register double result1_2 = result1_1[k];
			register double *Bt_1 = &Bt[k * N];
			for (j = 0; j < N; j++) {
				result2_1[j] += result1_2 * Bt_1[j];
			}
		}
	}

	
	for (i = 0; i < N; i++) {
		register double *C_1 = &C[i * N];
		register double *At_1 = &At[i * N];
		for (k = 0; k < N; k++) {
			if (k > i) {
				break;
			}
			register double *A_1 = &A[k * N];
			register double At_2 = At_1[k];
			for (j = 0; j < N; j++) {
				C_1[j] += At_2 * A_1[j];
			}
		}
	}

	
	for (i = 0; i < N; i++) {
		register double *C_1 = &C[i * N];
		register double *result2_1 = &result2[i * N];
		for (j = 0; j < N; j++) {
			C_1[j] += result2_1[j];
		}
	}

	free(At);
	free(Bt);
	free(result1);
	free(result2);

	return C;
}