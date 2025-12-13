
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MIN(a,b) (((a)<(b))?(a):(b))


double* my_solver(int N, double *A, double* B) {
	double *rez;

	double *prod11, *prod12;
	double *prod2;

	int i, j, k;

	rez = calloc(N * N, sizeof(double));
	if (rez == NULL)
		exit(EXIT_FAILURE);

	prod11 = calloc(N * N, sizeof(double));
	if (prod11 == NULL)
		exit(EXIT_FAILURE);

	prod12 = calloc(N * N, sizeof(double));
	if (prod12 == NULL)
		exit(EXIT_FAILURE);

	prod2 = calloc(N * N, sizeof(double));
	if (prod2 == NULL)
		exit(EXIT_FAILURE);

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = i; k < N; k++) {
				prod11[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				prod12[i * N + j] += prod11[i * N + k] * B[j * N + k];
			}
		}
	}

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k <= MIN(i, j); k++) {
				prod2[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			rez[i * N + j] = prod12[i * N + j] + prod2[i * N + j];
		}
	}

	free(prod11);
	free(prod12);
	free(prod2);
	return rez;
}
