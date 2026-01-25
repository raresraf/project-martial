
#include "utils.h"
#include <string.h>

void allocate_memory(int N, double **C, double **A_t, double **B_t,
					double **left_side, double **right_side, double **tmp)
{
	*C = calloc(N * N, sizeof(double));

	if (!(*C)) {
		exit(-1);
	}

	*A_t = calloc(N * N, sizeof(double));

	if (!(*A_t)) {
		exit(-1);
	}

	*B_t = calloc(N * N, sizeof(double));

	if (!(*B_t)) {
		exit(-1);
	}

	*left_side = calloc(N * N, sizeof(double));

	if (!(*left_side)) {
		exit(-1);
	}

	*right_side = calloc(N * N, sizeof(double));

	if (!(*right_side)) {
		exit(-1);
	}

	*tmp = calloc(N * N, sizeof(double));

	if (!(*tmp)) {
		exit(-1);
	}
}


double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	
	size_t i, j, k;
	double *C, *A_t, *B_t, *left_side, *right_side, *tmp;

	allocate_memory(N, &C, &A_t, &B_t, &left_side, &right_side, &tmp);

	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			*(A_t + N * i + j) = *(A + N * j + i);
			*(B_t + N * i + j) = *(B + N * j + i);
		}
	}

	
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			*(left_side + N * i + j) = 0.0;
			for (k = i; k < N; ++k) {
				
				*(left_side + N * i + j) += *(A + N * i + k) * *(B + N * k + j);
			}
		}
	}

	
	memcpy(tmp, left_side, N * N * sizeof(double));

	
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			*(left_side + N * i + j) = 0.0;
			for (k = 0; k < N; ++k) {
				*(left_side + N * i + j) += *(tmp + N * i + k) * *(B_t + N * k + j);
			}
		}
	}

	
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			*(right_side + N * i + j) = 0.0;
			for (k = 0; k < j + 1; ++k) {
				*(right_side + N * i + j) += *(A_t + N * i + k) * *(A + N * k + j);
			}
		}
	}

	
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			*(C + N * i + j) = *(left_side + N * i + j) + *(right_side + N * i + j);
		}
	}

	free(left_side);
	free(right_side);
	free(tmp);
	free(A_t);
	free(B_t);

	return C;
}
