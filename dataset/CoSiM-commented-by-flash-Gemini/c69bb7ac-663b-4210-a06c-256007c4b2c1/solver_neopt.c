
#include "utils.h"




void mult(int N, double *mat1, double *mat2, double *rez) {
	int i, j, k;

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				rez[i * N + j] += (mat1[i * N + k] * mat2[j + k * N]) ;
			}
		}
	}
}


void mult_triang_sup_norm(int N, double *mat1, double *mat2, double *rez) {
	int i, j, k;

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = i; k < N; k++) {
				rez[i * N + j] += (mat1[i * N + k] * mat2[k * N + j]) ;
			}
		}
	}
}


void mult_triang_inf_triang_sup(int N, double *mat1, double *mat2, double *rez) {
	int i, j, k;

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k <= (j < i ? j : i); k++) {
				rez[i * N + j] += mat1[i * N + k] * mat2[k * N + j] ;
			}
		}
	}
}


void add(int N, double *mat1, double *mat2) {
	int i, j;

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			mat1[i * N + j] += mat2[i * N + j];
		}
	}
}


void transp(int N, double *mat, int triang, double *rez) {
	int i, j;

	for (i = 0; i < N; ++i) {
		for (j = 0 + i * triang; j < N; ++j) {
			rez[i + N * j] = mat[i * N + j];
		}
	}
}

double* my_solver(int N, double *A, double* B) {
	double *rez, *A_t, *B_t, *aux1, *aux2;

	B_t = (double*) calloc(N * N, sizeof(double));
	A_t = (double*) calloc(N * N, sizeof(double));
	aux1 = (double*) calloc(N * N, sizeof(double));
	aux2 = (double*) calloc(N * N, sizeof(double));
	rez = (double*) calloc(N * N, sizeof(double));
	if (!B_t || !A_t || !aux1 || !aux2 || !rez)
		return NULL;

	
	transp(N, B, 0, B_t);

	
	transp(N, A, 1, A_t);

	
	mult_triang_sup_norm(N, A, B, aux1); 
	
	
	mult(N, aux1, B_t, rez);

	
	mult_triang_inf_triang_sup(N, A_t, A, aux2);

	
	add(N, rez, aux2);

	free(B_t);
	free(A_t);
	free(aux1);
	free(aux2);
	
	return rez;
}
