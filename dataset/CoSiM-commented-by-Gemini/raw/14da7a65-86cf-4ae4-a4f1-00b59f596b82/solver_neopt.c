
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	double *A_tA, *AB, *ABB_t, *C;
	int i, j, k;

	A_tA = (double *)calloc(N * N, sizeof(double));
	if (NULL == A_tA)
		exit(EXIT_FAILURE);
	
	AB = (double *)calloc(N * N, sizeof(double));
	if (NULL == AB)
		exit(EXIT_FAILURE);
	
	ABB_t = (double *)calloc(N * N, sizeof(double));
	if (NULL == ABB_t)
		exit(EXIT_FAILURE);
	
	C = (double *)calloc(N * N, sizeof(double));
	if (NULL == C)
		exit(EXIT_FAILURE);

	
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			for (k = i; k < N; k++)
				AB[i * N + j] += A[i * N + k] * B[k * N + j];

	
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			for (k = 0; k < N; k++)
				ABB_t[i * N + j] += AB[i * N + k] * B[j * N + k];

	
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			for (k = 0; k <= i; k++)
				A_tA[i * N + j] += A[k * N + i] * A[k * N + j];

	
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			C[i * N + j] = ABB_t[i * N + j] + A_tA[i * N + j];
	
	free(A_tA);
	free(AB);
	free(ABB_t);

	return C;
}
