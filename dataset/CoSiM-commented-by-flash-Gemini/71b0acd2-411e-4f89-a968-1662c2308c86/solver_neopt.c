
#include "utils.h"



void allocate_matrices(int N, double **C, double **BA_t, double **AA_t,
	double **AAB)
{
	*C = malloc(N * N * sizeof(**C));
	if (NULL == *C)
		exit(EXIT_FAILURE);

	*BA_t = calloc(N * N, sizeof(**BA_t));
	if (NULL == *BA_t)
		exit(EXIT_FAILURE);

	*AA_t = calloc(N * N, sizeof(**AA_t));
	if (NULL == *AA_t)
		exit(EXIT_FAILURE);

	*AAB = calloc(N * N, sizeof(**AAB));
	if (NULL == *AAB)
		exit(EXIT_FAILURE);
}

double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");

	double *C;
	double *BB_t;
	double *AA_t;
	double *ABB_t;

	int i, j, k;
	
	allocate_matrices(N, &C, &BB_t, &AA_t, &ABB_t);

	
	for (i = 0; i < N; i++) {
		for (j = 0; j <= i; j++) {
			BB_t[i * N + j] = 0.0;
			for (k = 0; k < N; k++) {
				BB_t[i * N + j] += B[i * N + k] * 
								B[j * N + k];
			}
			if(i != j) {
				BB_t[j * N + i] = BB_t[i * N + j];
			}
		}
	}

	
	for (i = 0; i < N; i++) {
		for (j = i; j < N; j++) {
			AA_t[i * N + j] = 0.0;
			for (k = 0; k <= i; k++) {
				AA_t[i * N + j] += A[k * N + i] * 
								A[k * N + j];
			}
			if(i != j) {
				AA_t[j * N + i] = AA_t[i * N + j];
			}
		}
	}
	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			ABB_t[(i * N + j)] = 0.0;
			for (k = i; k < N; k++) {
				ABB_t[(i * N + j)] += A[i * N + k] * 
								BB_t[k * N + j];
			}
		}
	}
	
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			C[i * N + j] = ABB_t[i * N + j] + AA_t[i * N + j];

	
	free(BB_t);
	free(AA_t);
	free(ABB_t);

	return C;
}
