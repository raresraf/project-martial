
#include "utils.h"

double* get_AxB(int N, double *A, double *B) {
	double *AxB = calloc(N * N, sizeof(double));
	int i, j, k;

	ASSERT(AxB == NULL, "malloc error");
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			for (k = i; k < N; k++)
				*(AxB + i * N + j) += *(A + i * N + k) * *(B + k * N + j);

	return AxB; 
}

double* get_AxBxBt(int N, double *A, double *B) {
	double *AxBxBt = calloc(N * N, sizeof(double));
	double *AxB = get_AxB(N, A, B);
	int i, j, k;

	ASSERT(AxBxBt == NULL, "malloc error");
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			for (k = 0; k < N; k++)
				*(AxBxBt + i * N + j) += *(AxB + i * N + k) * *(B + j * N + k);

	free(AxB);
	return AxBxBt;
}

double* get_AtxA(int N, double *A) {
	double *AtxA = calloc(N * N, sizeof(double));
	double *At = calloc(N * N, sizeof(double));
	int i, j, k;

	ASSERT(At == NULL, "malloc error");
	ASSERT(AtxA == NULL, "malloc error");
	for (i = 0; i < N; i++) {
		for (j = i; j < N; j++) {
			*(At + j * N + i) = *(A + i * N + j);
		}
	}

	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			for (k = 0; k <= j; k++)
				*(AtxA + i * N + j) += *(At + i * N + k) * *(A + k * N + j);

	free(At);
	return AtxA;
}

double* my_solver(int N, double *A, double* B) {
	double *AxBxBt = get_AxBxBt(N, A, B);
	double *AxAt = get_AtxA(N, A);
	double *ret = malloc(N * N * sizeof(double));
	int i, j;

	ASSERT(ret == NULL, "malloc error");
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			*(ret + i * N + j) = *(AxBxBt + i * N + j) + *(AxAt + i * N + j);

	free(AxBxBt);
	free(AxAt);
	return ret;
}
