
#include "utils.h"




void mult(int N, double *mat1, double *mat2, double *rez) {
	int i, j, k;

	for (i = 0; i < N; i++) {
		register int i_x_N = i * N;
		for (k = 0; k < N; k++) {
			register double ct = mat1[i_x_N + k];
			register int k_x_N = k * N;
			for (j = 0; j < N; j++) {
				rez[i_x_N + j] += ct * mat2[j + k_x_N];
			}
		}
	}
}


void mult_triang_inf_triang_sup(int N, double *mat1, double *mat2, double *rez) {
	int i, j, k;

	for (i = 0; i < N; i++) {
		register int i_x_N = i * N;
		for (j = 0; j < N; j++) {
			register double suma = 0.0;
			for (k = 0; k <= (j < i ? j : i); k++) {
				suma += mat1[i_x_N + k] * mat2[k * N + j] ;
			}
			rez[i_x_N + j] = suma;
		}
	}
}


void mult_triang_sup_norm(int N, double *mat1, double *mat2, double *rez) {
	int i, j, k;

	for (i = 0; i < N; i++) {
		register int i_x_N = i * N;
		for (k = i; k < N; k++) {
			register double ct = mat1[i_x_N + k];
			register int k_x_N = k * N;
			for (j = 0; j < N; j++) {
				rez[i_x_N + j] += ct * mat2[j + k_x_N];
			}
		}
	}
}


void add(int N, double *mat1, double *mat2) {
	int i, j;

	for (i = 0; i < N; i++) {
		register int i_x_N = i * N;
		for (j = 0; j < N; j++) {
			mat1[i_x_N + j] += mat2[i_x_N + j];
		}
	}
}


void transp(int N, double *mat, int triang, double *rez) {
	int i, j;

	for (i = 0; i < N; ++i) {
		register int i_x_N = i * N;
		for (j = 0 + i * triang; j < N; ++j) {
			rez[i + N * j] = mat[i_x_N + j];
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

