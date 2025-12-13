
#include "utils.h"



void alloc_matr(int N, double **C, double **AB, double **AA_t, double **ABB_t)
{
	*C = calloc(N * N, sizeof(**C));
	if (!*C)
		exit(EXIT_FAILURE);

	*AB = calloc(N * N, sizeof(**AB));
	if (!*AB)
		exit(EXIT_FAILURE);

	*AA_t = calloc(N * N, sizeof(**AA_t));
	if (!*AA_t)
		exit(EXIT_FAILURE);

	*ABB_t = calloc(N * N, sizeof(**ABB_t));
	if (!*ABB_t)
		exit(EXIT_FAILURE);
}

double* my_solver(int N, double *A, double* B)
{
	printf("NEOPT SOLVER\n");

	int i, j, k;

	double *AB = NULL;
	double *C = NULL;
	double *AA_t = NULL;
	double *ABB_t = NULL;

	alloc_matr(N, &C, &AB, &AA_t, &ABB_t);

	
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			for (k = i; k < N; ++k) {
				AB[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			for (k = 0; k < N; ++k) {
				ABB_t[i * N + j] += AB[i * N + k] * B[j * N + k];
			}
		}
	}

	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			for (k = 0; k <= j; ++k) {
				AA_t[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			C[i * N + j] += ABB_t[i * N + j] + AA_t[i * N + j];
		}
	}

	free(AB);
	free(AA_t);
	free(ABB_t);

	return C;
}
