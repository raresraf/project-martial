
#include "utils.h"




void add(int N, double *a, double *b, double *c) {

	int i;

	for (i = 0; i < N * N; i++) {
		c[i] = a[i] + b[i];
	}

}

void normal_x_normal_transpose(int N, double *a, double *c) {

	int i, j, k;
	
	double *pa = &a[0];

	for (i = 0; i < N; i++) {
		
		double *pa_t = &a[0];

		for (j = 0; j <= i; j++) {
			register double sum = 0.0;

			for (k = 0; k < N; k++) {
				
				sum += *(pa + k) * *(pa_t + k);
			}

			c[i * N + j] = sum;
			c[j * N + i] = sum;
			
			pa_t += N;
		}
		
		pa += N;
	}
}

void upper_x_normal(int N, double *a, double *b, double *c) {

	int i, j, k;
	double *pa = &a[0];
	double *pc = &c[0];

	for (i = 0; i < N; i++) {
		
		pa += i;
		
		double *pb = &b[i * N];

		for (k = i; k < N; k++) {
			register double ra = *pa;

			for (j = 0; j < N; j++) {
				
				*(pc + j) += ra * *pb;
				
				pb++;
			}
			
			pa++;
		}
		
		pc += N;
	}
}

void upper_transpose_x_upper(int N, double *a, double *c) {

	int i, j, k;
	
	double *orig_pa = &a[0];

	for (i = 0; i < N; i++) {

		for (j = 0; j <= i; j++) {
			
			register double sum = 0.0;
			register double *pa = orig_pa;

			for (k = 0; k <= j; k++) {
				
				sum += *(pa + i) * *(pa + j);
				
				pa += N;
			}

			c[i * N + j] = sum;
			c[j * N + i] = sum;
		}
		
	}
}

double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");

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
