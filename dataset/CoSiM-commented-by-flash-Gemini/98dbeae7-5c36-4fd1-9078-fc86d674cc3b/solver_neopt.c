
#include "utils.h"
#include "string.h"



void sum(double *c, double *b, int N) {
	int i, j;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			c[i * N + j] += b[i * N + j];
		}
	}
}

void mulAtA(double *c, double *a, double *b, int N) {
	int i, j, k;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k <= (i > j ? j : i); k++) {
				c[i * N + j] += a[k * N + i] * b[k * N + j];
			}
		}
	}
}

void mulBBt(double *c, double *a, double *b, int N) {
	int i, j, k;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				c[i * N + j] += a[i * N + k] * b[j * N + k];
			}
		}
	}
}

void mulABBt(double *c, double *a, double *b, int N) {
	int i, j, k;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = i; k < N; k++) {
				c[i * N + j] += a[i * N + k] * b[k * N + j];
			}
		}
	}
}

double* my_solver(int N, double *A, double* B) {
	double *C = (double*)calloc(N * N, sizeof(double));
	double *D = (double*)calloc(N * N, sizeof(double));
	double *E = (double*)calloc(N * N, sizeof(double));
	
	mulAtA(C, A, A, N);
	
	mulBBt(D, B, B, N);
	
	mulABBt(E, A, D, N);
	
	sum(C, E, N);
	free(D); free(E);
	return C;
}
