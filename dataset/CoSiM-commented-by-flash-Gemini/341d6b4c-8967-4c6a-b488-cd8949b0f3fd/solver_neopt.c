
#include "utils.h"




void add(int N, double *a, double *b, double *c) {
	int i;
	for (i = 0; i < N * N; i++) {
		c[i] = a[i] + b[i];
	}
}


void normal_x_normal_transpose(int N, double *a, double *c) {

	int i, j, k;

	for (i = 0; i < N; i++) {
		
		for (j = 0; j <= i; j++) {
			for (k = 0; k < N; k++) {
				
				c[i * N + j] += a[i * N + k] * a[j * N + k];
				
				c[j * N + i] = c[i * N + j];
			}
		}
	}
}



void upper_x_normal(int N, double *a, double *b, double *c) {

	int i, j, k;

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			
			for (k = i; k < N; k++) {
				c[i * N + j] += a[i * N + k] * b[k * N + j];
			}
		}
	}
}


void upper_transpose_x_upper(int N, double *a, double *c) {

	int i, j, k;

	for (i = 0; i < N; i++) {
		
		for (j = 0; j <= i; j++) {
			
			for (k = 0; k <= j; k++) {
				
				c[i * N + j] += a[k * N + i] * a[k * N + j];
				c[j * N + i] = c[i * N + j];
			}
		}
	}
}

double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");

	double *C = calloc(N * N, sizeof(double));
	double *BBt = calloc(N * N, sizeof(double));
	double *ABBt = calloc(N * N, sizeof(double));
	double *AtA = calloc(N * N, sizeof(double));

	
	normal_x_normal_transpose(N, B, BBt);
	upper_x_normal(N, A, BBt, ABBt);
	upper_transpose_x_upper(N, A, AtA);
	add(N, ABBt, AtA, C);

	free(BBt);
	free(ABBt);
	free(AtA);

	return C;
}
