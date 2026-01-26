
#include "utils.h"
#define UPPER 1
#define LOWER -1
#define NORMAL 0


void transpose(int N, double **C, double *M) {
	for(int i = 0; i < N; i ++) {
		for(int j = 0; j < N; j++) {
			(*C)[i * N + j] = M[j * N + i];
		}
	}
}

void multiply(int N, double **C, double *A, double *B, int typeA, int typeB) {
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
			for(int k = 0; k < N; k++) {
				if(typeA == NORMAL && typeB == NORMAL)
					(*C)[i * N + j] += A[i * N + k]* B[k * N + j];
				else if (typeA == UPPER && typeB == NORMAL) {
					if(i <= k)
						(*C)[i * N + j] += A[i * N + k]* B[k * N + j];
				} else if (typeA == LOWER && typeB == UPPER) {
					if(i >= k && k <= j)
						(*C)[i * N + j] += A[i * N + k]* B[k * N + j];
				}		
			}
		}
	}
}

void add(int N, double **C, double *A, double *B) {
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
			(*C)[i * N + j] = A[i * N + j] + B[i * N + j];
		}
	}
}


double* my_solver(int N, double *A, double* B) {
	double *AB = calloc(N* N,sizeof(double));
	multiply(N, &AB, A, B, UPPER, NORMAL);
	double *Btrans = malloc((N * N) * sizeof(double));
	transpose(N, &Btrans, B);
	double *ABBt = calloc(N* N,sizeof(double));
	multiply(N, &ABBt, AB, Btrans, NORMAL, NORMAL);

	double *Atrans = malloc((N * N) * sizeof(double));
	transpose(N, &Atrans, A);
	double *AtA = calloc(N* N,sizeof(double));
	multiply(N, &AtA, Atrans, A, LOWER, UPPER);

	double *result = malloc((N * N) * sizeof(double));
	add(N, &result, ABBt, AtA);

	free(AB);
	free(Btrans);
	free(ABBt);
	free(Atrans);
	free(AtA);

	return result;
}
