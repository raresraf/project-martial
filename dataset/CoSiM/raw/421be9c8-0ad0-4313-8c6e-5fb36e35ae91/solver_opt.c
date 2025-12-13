
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");

	double *C, *AB;
	int i, j, k;

	C = (double *)calloc(sizeof(double), N * N);
	AB = (double *)calloc(sizeof(double), N * N);

	for (i = 0; i < N; ++i) {
		register double *p1 = &A[i];
		for (j = 0; j < N; ++j) {
			register double *p2 = p1;
			register double *p3 = &A[j];
			register double sum = 0;
			for (k = 0; (k <= i && i < j) || (k <= j && i >= j); ++k) {
				sum += *p2 * *p3;
				p2 += N;
				p3 += N;
			}
			C[i * N + j] = sum;
		}
	}

	for (i = 0; i < N; ++i) {
		register double *p1 = &A[i * N + i];
		for (j = 0; j < N; ++j) {
			register double *p2 = p1;
			register double *p3 = &B[i * N + j];
			register double sum = 0;
			for (k = i; k < N; ++k) {
				sum += *p2 * *p3;
				p2++;
				p3 += N;
			}
			AB[i * N + j] = sum;
		}
	}

	for (i = 0; i < N; ++i) {
		register double *p1 = &AB[i * N];
		for (j = 0; j < N; ++j) {
			register double *p2 = p1;
			register double *p3 = &B[j * N];
			register double sum = 0;
			for (k = 0; k < N; ++k) {
				sum += *p2 * *p3;
				p2++;
				p3++;
			}
			C[i * N + j] += sum;
		}
	}

	free(AB);
	return C;
}
