
#include "utils.h"
#include <string.h>


static double *get_transpose(double *M, int N)
{
	double *tr = calloc(N * N, sizeof(double));
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			tr[i * N + j] = M[j * N + i];
		}
	}
	return tr;
}


double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");

	double *first_mul = calloc(N * N, sizeof(double));
	double *first_mul_aux = calloc(N * N, sizeof(double));
	double *second_mul = calloc(N * N, sizeof(double));
	double *Bt = get_transpose(B, N);
	double *At = get_transpose(A, N);

	
	for (int i = 0; i < N; ++i) {
		register double *aux = &A[i * N];
		for (int j = 0; j < N; ++j) {
			register double *collumn = &B[j];
			register double *line = aux;

			register double rez = 0;
			for (int k = i; k < N; ++k, line++, collumn += N) {
				rez += *(line + i) * *(collumn + i * N);
			}
			first_mul[i * N + j] = rez;
		}
	}


	
	for (int i = 0; i < N; ++i) {
		register double *aux = &first_mul[i * N];
		for (int j = 0; j < N; ++j) {
			register double *line = aux;
			register double *collumn = &Bt[j];
			register double res = 0;

			for (int k = 0; k < N; ++k, line++, collumn += N) {
				res += *line * *collumn;
			}
			first_mul_aux[i * N + j] = res;
		}
	}

		
	for (int i = 0; i < N; ++i) {
		register double *aux = &At[i * N];
		for (int j = 0; j < N; ++j) {
			register double *line = aux;
			register double *collumn = &A[j];
			register double res = 0;

			for (int k = 0; k <= i; ++k, line++, collumn += N) {
				res += *line * *collumn;
			}
			second_mul[i * N + j] = res;
		}
	}

	
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			first_mul_aux[i * N + j] += second_mul[i * N + j];
		}
	}

	
	free(first_mul);
	free(second_mul);
	free(At);
	free(Bt);

	return first_mul_aux;	
}
