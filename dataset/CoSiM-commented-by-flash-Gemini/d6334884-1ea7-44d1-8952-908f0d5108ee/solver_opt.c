
#include "utils.h"


double* compute3(register int N, register double *A, register double *ABBt) {
	register double *C = malloc(N * N * sizeof(double));
	register int i, j, k;

	
	
	for (i = 0; i < N; ++i) {
		register double *orig = &A[i];
		register double *offsetC = &C[i * N];
		register double *offsetABBt = &ABBt[i * N];
		for (j = 0; j < N; ++j) {
			register double *pa = orig;
			register double *pb = &A[j];
			register double sum = 0;
			for (k = 0; k <= i; ++k) {
				sum += *pa * *pb;
				pa += N;
				pb += N;
			}
			*offsetC = sum + *offsetABBt;
			offsetC++;
			offsetABBt++;
		}
	}

	return C;
}


double* compute2(register int N, register double *A, register double *B) {
	register double *C = malloc(N * N * sizeof(double));
	register int i, j, k;

	for (i = 0; i < N; ++i) {
		register double *orig = &A[i * N];
		for (j = 0; j < N; ++j) {
			register double *pa = orig;
			register double *pb = &B[j * N];
			register double sum = 0;
			for (k = 0; k < N; ++k) {
				sum += *pa * *pb;
				pa++;
				pb++;
			}
			C[i * N + j] = sum;
		}
	}

	return C;
}


double* compute1(register int N, register double *A, register double *B) {
	register double *C = malloc(N * N * sizeof(double));
	register int i, j, k;

	for (i = 0; i < N; ++i) {
		register double *orig = &A[i * N + i];
		for (j = 0; j < N; ++j) {
			register double *pa = orig;
			register double *pb = &B[i * N + j];
			register double sum = 0;
			for (k = i; k < N; ++k) {
				sum += *pa * *pb;
				pa++;
				pb += N;
			}
			C[i * N + j] = sum;
		}
	}

	return C;
}


double* my_solver(int N, double *A, double* B) {
	register double *AB = compute1(N, A, B);
	register double *ABBt = compute2(N, AB, B);
	register double *C = compute3(N, A, ABBt);

	free(AB);
	free(ABBt);

	return C;
}
