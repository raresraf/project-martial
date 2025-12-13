
#include "utils.h"


double* transpose_upper_matrix(int N, double *matrix) {
	int i, j;

	double *transpose = (double *)calloc(N * N, sizeof(double));
	if (transpose == NULL) {
		fprintf(stderr, "Error calloc transpose.\n");
		exit(0);
	}

	
	for (i = 0; i < N; i++) {
		for (j = i; j < N; j++) {
			transpose[j * N + i] = matrix[i * N + j];
		}
	}
	return transpose;
}


double *transpose_matrix(int N, double *matrix) {
	int i, j;

	double *transpose = (double *)calloc(N * N, sizeof(double));
	if (transpose == NULL) {
		fprintf(stderr, "Error calloc transpose.\n");
		exit(0);
	}

	
	for (i = 0; i < N; i ++) {
		for (j = 0; j < N; j++) {
			transpose[j * N + i] = matrix[i * N + j];
		}
	}
	return transpose;
}


double *compute_product_with_upper(int N, double *l, double *m) {
	int i, j, k;

	double *result = (double *)calloc(N * N, sizeof(double));
	if (result == NULL) {
		fprintf(stderr, "Error calloc product.\n");
		exit(0);
	}

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			result[i * N + j] = 0.0;

			for (k = i; k < N; k++) {
				result[i * N + j] += l[i * N + k] * m[k * N + j];
			}
		}
	}
	return result;
}


double *compute_product(int N, double *m1, double *m2) {
	int i, j, k;

	double *result = (double *)calloc(N * N, sizeof(double));
	if (result == NULL) {
		fprintf(stderr, "Error calloc product.\n");
		exit(0);
	}

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			result[i * N + j] = 0.0;

			for (k = 0; k < N; k++) {
				result[i * N + j] += m1[i * N + k] * m2[k * N + j]; 
			}
		}
	}

	return result;
}


double *compute_product_lower_upper(int N, double *l, double *u) {
	int i, j, k, min;

	double *result = (double *)calloc(N * N, sizeof(double));
	if (result == NULL) {
		fprintf(stderr, "Error calloc product.\n");
		exit(0);
	}

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			result[i * N + j] = 0.0;
			min = (i <= j) ? i : j;

			for (k = 0; k <= min; k++) {
				result[i * N + j] += l[i * N + k] * u[k * N + j];
			}
		}
	}

	return result;
}


double *compute_sum(int N, double *m1, double *m2) {
	int i, j;

	double *result = (double *)calloc(N * N, sizeof(double));
	if (result == NULL) {
		fprintf(stderr, "Error calloc sum.\n");
		exit(0);
	}

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			result[i * N + j] = m1[i * N + j] + m2[i * N + j];
		}
	}

	return result;
}



double* my_solver(int N, double *A, double* B) {
	
	
	double *At = transpose_matrix(N, A);

	
	double *Bt = transpose_matrix(N, B);

	
	double *prod_A_B = compute_product_with_upper(N, A, B);

	
	double *prod_Bt = compute_product(N, prod_A_B, Bt);

	
	double *prod_A_At = compute_product_lower_upper(N, At, A);

	
	double *sum_matrix = compute_sum(N, prod_Bt, prod_A_At);

	free(At);
	free(Bt);
	free(prod_A_B);
	free(prod_Bt);
	free(prod_A_At);

	return sum_matrix;
}
