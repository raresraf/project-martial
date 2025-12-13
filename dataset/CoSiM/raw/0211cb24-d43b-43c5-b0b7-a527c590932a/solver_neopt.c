
#include "utils.h"
#include <stdlib.h>

double* my_solver(int N, double *A, double *B) {
	int i, j, k;

	printf("NEOPT SOLVER\n");

	double *at = calloc(N * N, sizeof(double));
	if (at == NULL)
		exit(EXIT_FAILURE);

	double *bt = calloc(N * N, sizeof(double));
	if (bt == NULL)
		exit(EXIT_FAILURE);

	double *res1 = calloc(N * N, sizeof(double));
	if (res1 == NULL)
		exit(EXIT_FAILURE);

	double *res2 = calloc(N * N, sizeof(double));
	if (res2 == NULL)
		exit(EXIT_FAILURE);

	double *res3 = calloc(N * N, sizeof(double));
        if (res3 == NULL)
                exit(EXIT_FAILURE);

	double *res = calloc(N * N, sizeof(double));
        if (res == NULL)
                exit(EXIT_FAILURE);


	
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++) {
			at[j * N + i] = A[i *  N + j];
			bt[j * N + i] = B[i *  N + j];
		}

	
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			for (k = i; k < N; k++) {
				res1[i * N + j] += A[i * N + k]
					* B[k * N + j];
			}

	
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			for (k = 0; k < N; k++) {
				res2[i * N + j] += res1[i * N + k]
					* bt[k * N + j];
			}

	
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			for (k = 0; k <= j; k++) {
				res3[i * N + j] += at[i * N + k]
					* A[k * N + j];
			}

	
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++) {
			res[i * N + j] = res2[i * N + j] + res3[i * N + j];
		}

	free(at);
	free(bt);
	free(res1);
	free(res2);
	free(res3);
	return res;
}
