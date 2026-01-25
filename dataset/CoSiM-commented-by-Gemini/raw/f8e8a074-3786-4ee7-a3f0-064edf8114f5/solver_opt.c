
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	register int i, j, k;

	double *a_t = calloc(N * N, sizeof(*a_t));
	if (a_t == NULL) {
		exit(-1);
	}

	double *multiply = calloc(N * N, sizeof(*multiply));
	if (multiply == NULL)
		exit(-1);

	double *final_res = calloc(N * N, sizeof(*final_res));
	if (final_res == NULL)
		exit(-1);

	
	for (i = 0; i < N; i++) {
		register double *line_head = A + i * N;
		for (j = 0; j < N; j++) {
			register double *line = line_head + i;
			register double *col = B + j + i * N;
			register double sum = 0.0;

			for (k = i; k < N; k++, line++, col += N) {
				sum += *line * *col;
			}
 
			multiply[i * N + j] = sum;
		}
	}

	
	for (i = 0; i < N; i++) {
		register double *line_head = multiply + i * N;
		for (j = 0; j < N; j++) {
			register double *line = line_head;
			register double *col = B + j * N;
			register double sum = 0.0;

			for (k = 0; k < N; k++, line++, col++) {
				sum += *line * *col;
			}

			final_res[i * N + j] = sum;
		}
	}

	
	for (i = 0; i < N; i++) {
		register double *line_head = A + i;
		for (j = 0; j < N; j++) {
			register double *line = line_head;
			register double *col = A + j;
			register double sum = 0.0;

			for (k = 0; k <= i; k++, line += N, col += N) {
				sum += *line * *col;
			}

			a_t[i * N + j] = sum;
		}
	}

	
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			final_res[i * N + j] += a_t[i * N + j];
		}
	}

	free(a_t);
	free(multiply);

	return final_res;
}
