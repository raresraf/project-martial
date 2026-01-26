
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	double *RESULT = (double *) calloc(N * N, sizeof(double));
	double *TEMPORARY = (double *) calloc(N * N, sizeof(double));
	double *initial_line_parser, *line_parser, *column_parser;
	register double temporary_sum;
	register int i, j, k;

	
	for(i = 0; i < N; ++i){
		initial_line_parser = &A[i * N];
		for(j = 0; j < N; ++j){
			line_parser = initial_line_parser;
			column_parser = &B[j];
			temporary_sum = 0;
			for(k = 0; k < N; ++k, ++line_parser, column_parser += N){
				temporary_sum += *line_parser * *column_parser;
			}
			TEMPORARY[i * N + j] = temporary_sum;
		}
	}

	
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			temporary_sum = 0;
			for (k = 0; k < N; ++k) {
				temporary_sum += TEMPORARY[i * N + k] * B[j * N + k];
			}
			RESULT[i * N + j] += temporary_sum;
		}
	}

	
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			temporary_sum = 0;
			for (k = 0; k < N; ++k) {
				temporary_sum += A[k * N + i] * A[k * N + j];
			}
			RESULT[i * N + j] += temporary_sum;
		}
	}

	free(TEMPORARY);
	return RESULT;
}
