
#include "utils.h"
#include <string.h>


double *my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	size_t i, j, k;

	
	double *C = calloc(sizeof(double), N * N);
	double *transA = calloc(sizeof(double), N * N);
	double *transB = calloc(sizeof(double), N * N);

	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			*(transA + N * i + j) = *(A + N * j + i);
			*(transB + N * i + j) = *(B + N * j + i);
		}
	}

	
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			*(C + N * i + j) = 0.0;
			for (k = i; k < N; ++k) {
				*(C + N * i + j) += *(A + N * i + k) * *(B + N * k + j);
			}
		}
	}

	double *tmp = calloc(sizeof(double), N * N);
	memcpy(tmp, C, N * N * sizeof(double));

	
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			*(C + N * i + j) = 0.0;
			for (k = 0; k < N; ++k) {
				*(C + N * i + j) += *(tmp + N * i + k) * *(transB + N * k + j);
			}
		}
	}

	memcpy(tmp, C, N * N * sizeof(double));
	

	double *tmp2 = calloc(sizeof(double), N * N);
	
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			*(tmp2 + N * i + j) = 0.0;
			for (k = 0; k < N; ++k) {
				*(tmp2 + N * i + j) += *(transA + N * i + k) * *(A + N * k + j);
			}
		}
	}
	
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			*(C + N * i + j) = *(tmp + N * i + j) + *(tmp2 + N * i + j);
		}
	}

	free(tmp);
	free(tmp2);
	free(transA);
	free(transB);

	return C;
}
