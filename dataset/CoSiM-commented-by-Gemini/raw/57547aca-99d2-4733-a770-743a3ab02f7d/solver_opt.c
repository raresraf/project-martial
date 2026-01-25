#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	
	register int i = 0;
	register int j = 0;
	register int k = 0;

	
	double *At = malloc(N * N * sizeof(double));
	double *Bt = malloc(N * N * sizeof(double));

	for ( i = 0; i < N; i++) {
		for ( j = 0; j < N; j++) {
			register int index1 = i * N + j;
			register int index2 = j * N + i;

			At[index2] = A[index1];
			Bt[index2] = B[index1];
		}
	}

	double *AB = malloc(N * N * sizeof(double));
	
	for (i = 0; i < N; i++) {
		register int index = i * N;
		for (j = 0; j < N; j++) {
			register double sum = 0;
			for (k = i; k < N; k++) {
				sum += A[index + k] * B[k * N + j];
			}
			AB[index + j] = sum;
		}
	}

	
	double *AtA = malloc(N * N * sizeof(double));
	double *ABBt = malloc(N * N * sizeof(double));


	for (i = 0; i < N; i++) {
		for (k = 0; k < N; k++) {
			for (j = 0; j < N; j++) {
				AtA[i * N + j] += At[i * N + k] * A[k * N + j];
				ABBt[i * N + j] += AB[i * N + k] * Bt[k * N + j];
			}
		}
	}

	double *res = malloc(N * N * sizeof(double));

	for (i = 0; i < N; i++) {
		register int index = i * N;
		for (j = 0 ; j < N; j++) {
			res[index + j] = ABBt[index + j] + AtA[index + j];
		}
	}

	free(At);
	free(Bt);
	free(AtA);
	free(ABBt);
	printf("OPT SOLVER\n");
	return res;	
}

