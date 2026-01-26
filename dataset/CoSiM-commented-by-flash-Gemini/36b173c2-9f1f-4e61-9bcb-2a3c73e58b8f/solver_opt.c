
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
	int i, j, k;

	double *C = (double*) calloc(N * N, sizeof(double));
	if (C == NULL) {
		printf("Failed calloc!");
		return NULL;
	}

	double *another_C = (double*) calloc(N * N, sizeof(double));
	if (another_C == NULL) {
		printf("Failed calloc!");
		return NULL;
	}

	double *res_A = (double*) calloc(N * N, sizeof(double));
	if (res_A == NULL) {
		printf("Failed calloc!");
		return NULL;
	}

	
	for (i = 0; i < N; ++i) {
		register double *orig_pa = &A[i * N + i];
		for (j = 0; j < N; ++j) {
			register double *pa = orig_pa;
    			register double *pb = &B[i * N + j];
			register double sum = 0.0;
			for (k = 0; k < N - i; ++k) {
				sum += *pa * *pb;
				pa++;
				pb += N;
			}
			C[i * N + j] = sum;
		}
	}

	
	for (i = 0; i < N; ++i) {
		register double *orig_pc = &C[i * N + 0];
		for (j = 0; j < N; ++j) {
			register double *pc = orig_pc;
    		register double *pb = &B[j * N + 0];
			register double sum = 0.0;
			for (k = 0; k < N; ++k) {
				sum += *pc * *pb;
				pc++;
				pb++;
			}
			another_C[i * N + j] = sum;
		}
	}

	
	for (k = 0; k < N; ++k) {
		register double *pa = &A[k * N + k];
		for (i = k; i < N; ++i) {
			register double *pa_t = &A[k * N + k];
			for (j = 0; j < N - k; ++j) {
				res_A[i * N + k + j] += *pa_t * *pa;
				pa_t++;
			}
			pa++;
		}
	}

	
	for (i = 0; i < N; ++i) {
		register double *orig_pa = &res_A[i * N + 0];
		for (j = 0; j < N; ++j) {
			another_C[i * N + j] += *orig_pa;
			orig_pa++;
		}
	}
	free(res_A);
	free(C);
	return another_C;	
}
