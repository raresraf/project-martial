
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
	register int i, j, k;
	
	double *temp = (double *) calloc(N * N, sizeof(double));
	double *temp1 = (double *) calloc(N * N, sizeof(double));
	double *temp2 = (double *) calloc(N * N, sizeof(double));
	
	double *res = (double *) calloc(N * N, sizeof(double));

	
	for (i = 0; i < N; ++i) {
		register double *A_t_ptr = temp + i;
                register double *B_t_ptr = temp1 + i;
                register double *A_ptr = A + i * N;
		register double *B_ptr = B + i * N;
		for (j = 0; j < N; ++j, A_t_ptr += N, B_t_ptr += N, ++A_ptr, ++B_ptr) {
			*A_t_ptr = *A_ptr;
			*B_t_ptr = *B_ptr;
		}
	}

	
	for (i = 0; i < N; ++i) {
		register double *temp_copy = temp + i * N;

		for (j = 0; j < N; ++j) {
			register double result = 0;
			register double *A_t_ptr = temp_copy;
			register double *A_ptr = A + j;

			for (k = 0; k <= i; ++k, ++A_t_ptr, A_ptr += N) 
				result += *A_t_ptr * *A_ptr;

			res[i * N + j] = result;
		}
	}
	
	
	for (i = 0; i < N; ++i) {
		register double *A_copy = A + i * N;
		for (j = 0; j < N; ++j) {
			register double result = 0;
			register double *A_ptr = A_copy;
			register double *B_ptr = B + j;

			for (k = 0; k < N; ++k, ++A_ptr, B_ptr += N) {
				result += *A_ptr * *B_ptr;
			}	

			temp2[i * N + j] = result;
		}
	}
	
	for (i = 0; i < N; ++i) {
		register double *temp2_copy = temp2 + i * N;

		for (j = 0; j < N; ++j) {
			register double result = 0;
			register double *temp2_ptr = temp2_copy;
			register double *B_t_ptr = temp1 + j;

			for (k = 0; k < N; ++k, ++temp2_ptr, B_t_ptr += N) {
				result += *temp2_ptr * *B_t_ptr;
			}

			res[i * N + j] += result;
		}
	}

	free(temp);
	free(temp1);
	free(temp2);
	
	return res;	
}

