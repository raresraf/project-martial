
#include "utils.h"
#include <string.h>


double* my_solver(int N, double *A, double* B) {
	register double *ab = (double *) malloc(N * N * sizeof(double));
	register double *abbt = (double *) malloc(N * N * sizeof(double));
	register double *ata = (double *) malloc(N * N * sizeof(double));
	register double *result =  (double *) malloc(N * N * sizeof(double));

	
	for (int i = 0; i < N; ++i) {
		register double *a_i = A + i * N;
		for (int j = 0; j < N; ++j) {			
			register double *aux_a_i = a_i;
			register double *b_j = B + j;
			register double sum = 0.0;

			b_j += i * N;

			for (int k = i; k < N; ++k) {
				sum += *(aux_a_i + i) * *b_j;
				aux_a_i++;
				b_j += N;
			}
			ab[i * N + j] = sum;
		}
	}

	
	for (int i = 0; i < N; ++i) {
		register double *ab_in = ab + i * N;
		for (int j = 0; j < N; ++j) {
			register double *aux_ab_in = ab_in;
			register double *b_jn = B + j * N;
			register double sum = 0.0;

			for (int k = 0; k < N; ++k) {
				sum += *aux_ab_in * *b_jn;
				++aux_ab_in;
				++b_jn;
			}
			abbt[i * N + j] = sum;
		}
	}

	
	for (int i = 0; i < N; ++i) {
		register double *a_i = A + i;
		for (int j = 0; j < N; ++j) {
			register double *aux_a_i = a_i;
			register double *aux_a_j = A + j;
			register double sum = 0.0;
			
			for (int k = 0; k <= i; ++k) {
				sum += *aux_a_i * *aux_a_j;
				aux_a_i += N;
				aux_a_j += N;
			}
			ata[i * N + j] = sum;
		}
	}

	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			result[i * N + j] = abbt[i * N + j] + ata[i * N + j];
		}
	}
	
	free(ab);
	free(abbt);
	free(ata);
	return result;	
}
