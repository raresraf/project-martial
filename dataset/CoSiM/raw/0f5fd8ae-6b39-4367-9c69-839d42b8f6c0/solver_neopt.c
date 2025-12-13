
#include "utils.h"




double* AB(int N, double *A, double *B) {
	int i, j, k;
	double *AB;
	
	AB = calloc(N * N, sizeof(double));
	if (!AB) {
		exit(EXIT_FAILURE);
	}

	
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			for (k = i; k < N; k++)
				AB[i * N + j] += A[i * N + k] * B[k * N + j];

	return AB;
}


double* ABB_t(int N, double *AB, double *B) {
	int i, j, k;
	double *ABB_t;
	
	ABB_t = calloc(N * N, sizeof(double));
	if (!ABB_t) {
		exit(EXIT_FAILURE);
	}

	
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			for (k = 0; k < N; k++)
				ABB_t[i * N + j] += AB[i * N + k] * B[j * N + k];

	return ABB_t;
}


double* A_tA(int N, double *A_t, double *A) {
	int i, j, k;
	double *A_tA;
	
	A_tA = calloc(N * N, sizeof(double));
	if (!A_tA) {
		exit(EXIT_FAILURE);
	}

	
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			for (k = 0; k < i + 1; k++)
				A_tA[i * N + j] += A_t[k * N + i] * A[k * N + j];

	return A_tA;
}


double* sum(int N, double *A, double *B) {
	int i, j;
	double *C;
	
	C = calloc(N * N, sizeof(double));
	if (!C) {
		exit(EXIT_FAILURE);
	}

	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			C[i * N + j] = A[i * N + j] + B[i * N + j];

	return C;
}

double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");

	double *ptr_AB = AB(N, A, B);
	double *ptr_ABB_t = ABB_t(N, ptr_AB, B);
	double *ptr_A_tA = A_tA(N, A, A);
	double *ptr_sum = sum(N, ptr_ABB_t, ptr_A_tA);

	free(ptr_AB);
	free(ptr_ABB_t);
	free(ptr_A_tA);

	return ptr_sum;
}
