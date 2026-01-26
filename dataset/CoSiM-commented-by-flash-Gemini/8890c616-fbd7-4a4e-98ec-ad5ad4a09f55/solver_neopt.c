
#include <stdlib.h>
#include "utils.h"


void allocate(int N, double **C, double **AB, double **ABB_t,
			  double **A_tA)
{
	*C = malloc(N * N * sizeof(**C));
	if (NULL == *C)
		exit(EXIT_FAILURE);

	*AB = calloc(N * N, sizeof(**AB));
	if (NULL == *AB)
		exit(EXIT_FAILURE);

	*ABB_t = calloc(N * N, sizeof(**ABB_t));
	if (NULL == *ABB_t)
		exit(EXIT_FAILURE);

	*A_tA = calloc(N * N, sizeof(**A_tA));
	if (NULL == *A_tA)
		exit(EXIT_FAILURE);
}


double* my_solver(int N, double *A, double* B)
{
	double *C, *AB, *ABB_t, *A_tA;

	int i, j, k;

	allocate(N, &C, &AB, &ABB_t, &A_tA);

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = i; k < N; k++) {
				AB[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				ABB_t[i * N + j] += AB[i * N + k] * B[j * N + k];
			}
		}
	}

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			if (i < j) {
				for (k = 0; k <= i; k++) {
					A_tA[i * N + j] += A[k * N + i] * A[k * N + j];
				}
			} else {
				for (k = 0; k <= j; k++) {
					A_tA[i * N + j] += A[k * N + i] * A[k * N + j];
				}
			}
		}
	}

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i * N + j] = ABB_t[i * N + j] + A_tA[i * N + j];
		}
	}

	
	free(AB);
	free(ABB_t);
	free(A_tA);

	return C;
}
