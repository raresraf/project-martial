
#include "utils.h"
#include "string.h"



void sum(double *c, double *b, int N) {
	int i;
	for (i = 0; i < N * N; i++) {
		*c += *b;
		c++;
		b++;
	}
}

void mulAtA(double *c, double *a, double *b, int N) {
	int i, j, k;
	for (j = 0; j < N; j++) {
		for (k = 0; k <= j; k++) {
			register int row_k = k * N;
			register double *pointer = &(c[j]);
			register double *elem_b = &(b[row_k + j]);
			register double *elem_a = &(a[row_k]);
			for (i = 0; i < N; i++) {
				*pointer += *elem_a * *elem_b;
				pointer += N;
				elem_a ++;
			}
		}
	}
}

void mulBBt(double *c, double *a, double *b, int N) {
	int i, j, k;
	double sum;
	for (i = 0; i < N; i++) {
		register int row_i = i * N;
		for (j = 0; j < N; j++) {
			register int row_j = j * N;
			register double* elem_a = &(a[row_i]);
			register double* elem_b = &(b[row_j]);
			sum = 0;
			for (k = 0; k < N; k++) {
				sum += *elem_a * *elem_b;
				elem_a++;
				elem_b++;
			}
			*c = sum;
			c++;
		}
	}
}

void mulABBt(double *c, double *a, double *b, int N) {
	int i, j, k;
	for (i = 0; i < N; i++) {
		register int row_i = i * N;
		for (j = 0; j < N; j++) {
			double sum = 0;
			register double* elem_a = &(a[row_i + i]);
			register double* elem_b = &(b[row_i + j]);
			for (k = i; k < N; k++) {
				sum += *elem_a * *elem_b;
				elem_a++;
				elem_b += N;
			}
			*c = sum;
			c++;
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
