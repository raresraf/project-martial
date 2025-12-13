
#include "utils.h"
#include <string.h>


double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	double *C, *AB, *right;
	int i, j, k;
	C =(double *)calloc(N * N, sizeof(double));
	if (C == NULL)
		exit(-1);
	AB =(double *)calloc(N * N, sizeof(double));
	if (AB == NULL)
		exit(-1);
	right =(double *)calloc(N * N, sizeof(double));
	if (right == NULL)
		exit(-1);

	
	for (i = 0; i < N; ++i)
		for (j = 0; j < N; ++j)
			for (k = i; k < N; ++k)
				AB[i * N + j] += A[i * N + k] * B[k * N + j];

	
	for (i = 0; i < N; ++i)
		for (j = 0; j < N; ++j)
			for (k = 0; k < N; ++k)
				C[i * N + j] += AB[i * N + k] * B[j * N + k];

	
	for (i = 0; i < N; ++i)
		for (j = 0; j < N; ++j)
			for (k = 0; k <= i; ++k)
				right[i * N + j] += A[k * N + i] * A[k * N + j];

	
	for (i = 0; i < N; ++i)
		for (j = 0; j < N; ++j)
			C[i * N + j] += right[i * N + j];

	free(right);
	free(AB);
	return C;
}
