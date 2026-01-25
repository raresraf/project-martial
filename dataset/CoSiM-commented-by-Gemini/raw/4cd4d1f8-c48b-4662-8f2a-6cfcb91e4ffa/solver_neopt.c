
#include "utils.h"

double* my_solver(int N, double *A, double* B)
{
	double *AtA, *C, *ABBt, *AB;

	
	C = malloc(N * N * sizeof(*C));
	if (NULL == C)
		exit(1);

	AtA = malloc(N * N * sizeof(*AtA));
	if (NULL == AtA)
		exit(1);

	AB = malloc(N * N * sizeof(*AB));
	if (NULL == AB)
		exit(1);

	ABBt = malloc(N * N * sizeof(*ABBt));
	if (NULL == ABBt)
		exit(1);
	
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++) {
			AB[i * N + j] = 0;
			ABBt[i * N + j] = 0;
			AtA[i * N + j] = 0;
			C[i * N + j] = 0;
		}

	
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			AB[i * N + j] = 0;
			for (int k = i; k < N; k++){
				AB[i * N + j] += A[i * N + k]* B[k * N + j];
			}
		}
	}
	
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			ABBt[i * N + j] = 0;
			for (int k = 0; k < N; k++) {
				ABBt[i * N + j] += AB[i * N + k] * B[j * N + k];
			}
		}
	}

	
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			AtA[i * N + j] = 0;
			for (int k = 0; k <= j; k++){
				AtA[i * N + j] += A[k * N + i]* A[k * N + j];
			}
		}
	}
	
	
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			C[i * N + j] = ABBt[i * N + j] + AtA[i * N + j];
		}
	}

	free(AB);
	free(ABBt);
	free(AtA);
	return C;
}
