/**
 * @file solver_neopt.c
 * @brief Semantic documentation for solver_neopt.c. 
 *        This is a placeholder. Detailed semantic analysis will be applied later.
 */

#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	double *C;
	double *ABB_t;
	double *A_tA;
	double *AB;
	int i, j, k;

	C = malloc(N * N * sizeof(*C));
	if (NULL == C)
		exit(EXIT_FAILURE);

	ABB_t = calloc(N * N, sizeof(*ABB_t));
	if (NULL == ABB_t)
		exit(EXIT_FAILURE);

	A_tA = calloc(N * N, sizeof(*A_tA));
	if (NULL == A_tA)
		exit(EXIT_FAILURE);

	AB = calloc(N * N, sizeof(*AB));
	if (NULL == AB)
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
			for (k = 0; k < N; k++)
				A_tA[i * N + j] += A[k * N + i] * A[k * N + j];

	
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			C[i * N + j] = ABB_t[i * N + j] + A_tA[i * N + j];

	free(ABB_t);
	free(A_tA);
	free(AB);

	return C;
}
