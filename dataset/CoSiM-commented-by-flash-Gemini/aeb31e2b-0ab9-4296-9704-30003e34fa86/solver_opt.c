
#include <string.h>
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
	double *C, *aux;
	register double *first, *second, *third;
	int i, j, k;
	C = (double *)malloc(N * N * sizeof(double));
	aux = (double *)malloc(N * N * sizeof(double));
	memset(C, 0, N * N * sizeof(double));
	memset(aux, 0, N * N * sizeof(double));

	for (i = 0; i < N; ++i) {
		for (k = i; k < N; ++k) {
			first = A + i * N + k;
			second = B + k * N;
			third = aux + i * N;
			
			for (j = 0; j < N; ++j) {
				*third += *first * *second;
				second++;
				third++;
			}

			first++;
		}
	}

	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			first = aux + i * N;
			second = B + j * N;
			register double sum = 0.0;
			
			for (k = 0; k < N; ++k) {
				sum += *first * *second;
				first++;
				second++;
			}

			C[i * N + j] = sum;
		}
	}

	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			first = A + i;
			second = A + j;
			register double sum = 0.0;
			register int lim;

			if (i < j)
				lim = i;
			else
				lim = j;
			
			for (k = 0; k <= lim; ++k) {
				sum += *first * *second;
				first += N;
				second += N;
			}

			C[i * N + j] += sum;
		}
	}

	free(aux);

	return C;
}
