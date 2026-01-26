
#include "utils.h"

double* get_AxB(int N, double *A, double *B) {
	register double *AxB = calloc(N * N, sizeof(double));
	register int i;
	register int j;
	register int k;

	ASSERT(AxB == NULL, "malloc error");
	for (i = 0; i < N; ++i) {
		
		register double *c_axb = AxB + i * N;
		
		register double *c_a = A + i * N;

		for (j = 0; j < N; ++j, ++c_axb) {
			register double sum = 0;
			register double *p_b = B + j + i * N;
			register double *p_a = c_a + i;

			for (k = i; k < N; ++k, ++p_a, p_b += N)
				sum += *p_a * *p_b;

			*c_axb = sum;
		}
	}

	return AxB; 
}

double* get_AxBxBt(int N, double *A, double *B) {
	register double *AxBxBt = calloc(N * N, sizeof(double));
	register double *AxB = get_AxB(N, A, B);
	register int i;
	register int j;
	register int k;

	ASSERT(AxBxBt == NULL, "malloc error");
	for (i = 0; i < N; ++i) {
		
		register double *c_axbxbt = AxBxBt + i * N;

		for (j = 0; j < N; ++j, ++c_axbxbt) {
			register double sum = 0;
			register double *p_axb = AxB + i * N;
			register double *p_b = B + j * N;

			for (k = 0; k < N; ++k, ++p_b, ++p_axb)
				sum += *p_b * *p_axb;

			*c_axbxbt = sum;
		}
	}

	free(AxB);
	return AxBxBt;
}

double* get_AtxA(int N, double *A) {
	register double *AtxA = calloc(N * N, sizeof(double));
	double *At = calloc(N * N, sizeof(double));
	register int i;
	register int j;
	register int k;

	ASSERT(At == NULL, "malloc error");
	ASSERT(AtxA == NULL, "malloc error");
	for (i = 0; i < N; i++) {
		for (j = i; j < N; j++) {
			*(At + j * N + i) = *(A + i * N + j);
		}
	}

	for (i = 0; i < N; ++i) {
		
		register double *c_axat = AtxA + i * N;

		for (j = 0; j < N; ++j, ++c_axat) {
			register double sum = 0;
			register double *p_a = A + j;
			register double *p_at = At + i * N;

			for (k = 0; k <= j; ++k, p_a += N, ++p_at)
				sum += *p_a * *p_at;

			*c_axat = sum;
		}
	}

	free(At);
	return AtxA;
}

double* my_solver(int N, double *A, double* B) {
	register double *AxBxBt = get_AxBxBt(N, A, B);
	register double *AxAt = get_AtxA(N, A);
	register double *ret = malloc(N * N * sizeof(double));
	register int i;
	register int j;

	ASSERT(ret == NULL, "malloc error");
	for (i = 0; i < N; ++i) {
		register double *c_ret = ret + i * N;
		register double *c_axabt = AxBxBt + i * N;
		register double *c_axat = AxAt + i * N;

		for (j = 0; j < N; ++j, ++c_axat, ++c_axabt, ++c_ret) {
			*c_ret = *c_axabt + *c_axat;
		}
	}		

	free(AxBxBt);
	free(AxAt);
	return ret;
}
