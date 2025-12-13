
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	int i, j, k;
	double *M1 = (double *)calloc(N * N, sizeof(double)); 
	double *M2 = (double *)calloc(N * N, sizeof(double)); 
	double *M3 = (double *)calloc(N * N, sizeof(double)); 
	if (!M1 || !M2 || !M3)
		return NULL;

	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			for(k = i; k < N; k++)
				M1[i * N + j] += A[i * N + k] * B[k * N + j];

	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			for(k = 0; k < N; k++)
			 	M2[i * N + j] += M1[i * N + k] * B[j * N + k];

	int end;
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++) {
			if (i < j)
				end = i;
			else
				end = j;
			for(k = 0; k <= end; k++)
			 	M3[i * N + j] += A[k * N + i] * A[k * N + j];
			}

	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			M2[i * N + j] += M3[i * N + j];

	free(M1);
	free(M3);
	return M2;
}
