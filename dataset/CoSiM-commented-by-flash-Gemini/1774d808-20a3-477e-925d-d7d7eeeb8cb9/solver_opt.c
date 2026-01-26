
#include "utils.h"


double* transpose_upper_matrix(int N, double *matrix) {
	int i, j;

	double *transpose = (double *)calloc(N * N, sizeof(double));
	if (transpose == NULL) {
		fprintf(stderr, "Error calloc transpose.\n");
		exit(0);
	}

	
	
	for (i = 0; i < N; i++) {
		register double *m_elem = &matrix[i * N];

		for (j = i; j < N; j++) {
			transpose[j * N + i] = m_elem[j];
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
		register double *m_elem = &matrix[i * N];

		for (j = 0; j < N; j++) {
			transpose[j * N + i] = m_elem[j];
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
		register double *result_elem = &result[i * N];
		register double *lower_elem = &l[i * N];

		for (k = i; k < N; k++) {
			register double l_elem = lower_elem[k];
			register double *m_elem = &m[k * N];

			for (j = 0; j < N; j += 40) {
				result_elem[j] += l_elem * m_elem[j];
				result_elem[j + 1] += l_elem * m_elem[j + 1];
				result_elem[j + 2] += l_elem * m_elem[j + 2];
				result_elem[j + 3] += l_elem * m_elem[j + 3];
				result_elem[j + 4] += l_elem * m_elem[j + 4];
				result_elem[j + 5] += l_elem * m_elem[j + 5];
				result_elem[j + 6] += l_elem * m_elem[j + 6];
				result_elem[j + 7] += l_elem * m_elem[j + 7];
				result_elem[j + 8] += l_elem * m_elem[j + 8];
				result_elem[j + 9] += l_elem * m_elem[j + 9];
				result_elem[j + 10] += l_elem * m_elem[j + 10];
				result_elem[j + 11] += l_elem * m_elem[j + 11];
				result_elem[j + 12] += l_elem * m_elem[j + 12];
				result_elem[j + 13] += l_elem * m_elem[j + 13];
				result_elem[j + 14] += l_elem * m_elem[j + 14];
				result_elem[j + 15] += l_elem * m_elem[j + 15];
				result_elem[j + 16] += l_elem * m_elem[j + 16];
				result_elem[j + 17] += l_elem * m_elem[j + 17];
				result_elem[j + 18] += l_elem * m_elem[j + 18];
				result_elem[j + 19] += l_elem * m_elem[j + 19];
				result_elem[j + 20] += l_elem * m_elem[j + 20];
				result_elem[j + 21] += l_elem * m_elem[j + 21];
				result_elem[j + 22] += l_elem * m_elem[j + 22];
				result_elem[j + 23] += l_elem * m_elem[j + 23];
				result_elem[j + 24] += l_elem * m_elem[j + 24];
				result_elem[j + 25] += l_elem * m_elem[j + 25];
				result_elem[j + 26] += l_elem * m_elem[j + 26];
				result_elem[j + 27] += l_elem * m_elem[j + 27];
				result_elem[j + 28] += l_elem * m_elem[j + 28];
				result_elem[j + 29] += l_elem * m_elem[j + 29];
				result_elem[j + 30] += l_elem * m_elem[j + 30];
				result_elem[j + 31] += l_elem * m_elem[j + 31];
				result_elem[j + 32] += l_elem * m_elem[j + 32];
				result_elem[j + 33] += l_elem * m_elem[j + 33];
				result_elem[j + 34] += l_elem * m_elem[j + 34];
				result_elem[j + 35] += l_elem * m_elem[j + 35];
				result_elem[j + 36] += l_elem * m_elem[j + 36];
				result_elem[j + 37] += l_elem * m_elem[j + 37];
				result_elem[j + 38] += l_elem * m_elem[j + 38];
				result_elem[j + 39] += l_elem * m_elem[j + 39];	
			}
		}
	}
	return result;
}


double *compute_product(int N, double *m1, double *m2) {
	int i, j, k;

	double *result = (double *)calloc(N * N, sizeof(double));
	if (result == NULL) {
		fprintf(stderr, "Error calloc transpose.\n");
		exit(0);
	}

	
	for (i = 0; i < N; i++) {
		register double *result_elem = &result[i * N];
		register double *m1_elem = &m1[i * N];

		for (k = 0; k < N; k++) {
			register double *m2_elem = &m2[k * N];
			register double m1_el = m1_elem[k];

			for (j = 0; j < N; j += 40) {
				result_elem[j] += m1_el * m2_elem[j];
				result_elem[j + 1] += m1_el * m2_elem[j + 1];
				result_elem[j + 2] += m1_el * m2_elem[j + 2];
				result_elem[j + 3] += m1_el * m2_elem[j + 3];
				result_elem[j + 4] += m1_el * m2_elem[j + 4];
				result_elem[j + 5] += m1_el * m2_elem[j + 5];
				result_elem[j + 6] += m1_el * m2_elem[j + 6];
				result_elem[j + 7] += m1_el * m2_elem[j + 7];
				result_elem[j + 8] += m1_el * m2_elem[j + 8];
				result_elem[j + 9] += m1_el * m2_elem[j + 9];
				result_elem[j + 10] += m1_el * m2_elem[j + 10];
				result_elem[j + 11] += m1_el * m2_elem[j + 11];
				result_elem[j + 12] += m1_el * m2_elem[j + 12];
				result_elem[j + 13] += m1_el * m2_elem[j + 13];
				result_elem[j + 14] += m1_el * m2_elem[j + 14];
				result_elem[j + 15] += m1_el * m2_elem[j + 15];
				result_elem[j + 16] += m1_el * m2_elem[j + 16];
				result_elem[j + 17] += m1_el * m2_elem[j + 17];
				result_elem[j + 18] += m1_el * m2_elem[j + 18];
				result_elem[j + 19] += m1_el * m2_elem[j + 19];
				result_elem[j + 20] += m1_el * m2_elem[j + 20];
				result_elem[j + 21] += m1_el * m2_elem[j + 21];
				result_elem[j + 22] += m1_el * m2_elem[j + 22];
				result_elem[j + 23] += m1_el * m2_elem[j + 23];
				result_elem[j + 24] += m1_el * m2_elem[j + 24];
				result_elem[j + 25] += m1_el * m2_elem[j + 25];
				result_elem[j + 26] += m1_el * m2_elem[j + 26];
				result_elem[j + 27] += m1_el * m2_elem[j + 27];
				result_elem[j + 28] += m1_el * m2_elem[j + 28];
				result_elem[j + 29] += m1_el * m2_elem[j + 29];
				result_elem[j + 30] += m1_el * m2_elem[j + 30];
				result_elem[j + 31] += m1_el * m2_elem[j + 31];
				result_elem[j + 32] += m1_el * m2_elem[j + 32];
				result_elem[j + 33] += m1_el * m2_elem[j + 33];
				result_elem[j + 34] += m1_el * m2_elem[j + 34];
				result_elem[j + 35] += m1_el * m2_elem[j + 35];
				result_elem[j + 36] += m1_el * m2_elem[j + 36];
				result_elem[j + 37] += m1_el * m2_elem[j + 37];
				result_elem[j + 38] += m1_el * m2_elem[j + 38];
				result_elem[j + 39] += m1_el * m2_elem[j + 39];
			}
		}
	}

	return result;
}


double *compute_product_lower_upper(int N, double *l, double *u) {
	int i, j, k;

	double *result = (double *)calloc(N * N, sizeof(double));
	if (result == NULL) {
		fprintf(stderr, "Error calloc product.\n");
		exit(0);
	}

	
	for (i = 0; i < N; i++) {
		register double *result_elem = &result[i * N];
		register double *l_elem = &l[i * N];

		for (k = 0; k < N; k++) {
			register double *upper_elem = &u[k * N];
			register double lower = l_elem[k];

			if (k <= i) {
				for (j = 0; j < N; j += 40) {
					result_elem[j] += lower * upper_elem[j];
					result_elem[j + 1] += lower * upper_elem[j + 1];
					result_elem[j + 2] += lower * upper_elem[j + 2];
					result_elem[j + 3] += lower * upper_elem[j + 3];
					result_elem[j + 4] += lower * upper_elem[j + 4];
					result_elem[j + 5] += lower * upper_elem[j + 5];
					result_elem[j + 6] += lower * upper_elem[j + 6];
					result_elem[j + 7] += lower * upper_elem[j + 7];
					result_elem[j + 8] += lower * upper_elem[j + 8];
					result_elem[j + 9] += lower * upper_elem[j + 9];
					result_elem[j + 10] += lower * upper_elem[j + 10];
					result_elem[j + 11] += lower * upper_elem[j + 11];
					result_elem[j + 12] += lower * upper_elem[j + 12];
					result_elem[j + 13] += lower * upper_elem[j + 13];
					result_elem[j + 14] += lower * upper_elem[j + 14];
					result_elem[j + 15] += lower * upper_elem[j + 15];
					result_elem[j + 16] += lower * upper_elem[j + 16];
					result_elem[j + 17] += lower * upper_elem[j + 17];
					result_elem[j + 18] += lower * upper_elem[j + 18];
					result_elem[j + 19] += lower * upper_elem[j + 19];
					result_elem[j + 20] += lower * upper_elem[j + 20];
					result_elem[j + 21] += lower * upper_elem[j + 21];
					result_elem[j + 22] += lower * upper_elem[j + 22];
					result_elem[j + 23] += lower * upper_elem[j + 23];
					result_elem[j + 24] += lower * upper_elem[j + 24];
					result_elem[j + 25] += lower * upper_elem[j + 25];
					result_elem[j + 26] += lower * upper_elem[j + 26];
					result_elem[j + 27] += lower * upper_elem[j + 27];
					result_elem[j + 28] += lower * upper_elem[j + 28];
					result_elem[j + 29] += lower * upper_elem[j + 29];
					result_elem[j + 30] += lower * upper_elem[j + 30];
					result_elem[j + 31] += lower * upper_elem[j + 31];
					result_elem[j + 32] += lower * upper_elem[j + 32];
					result_elem[j + 33] += lower * upper_elem[j + 33];
					result_elem[j + 34] += lower * upper_elem[j + 34];
					result_elem[j + 35] += lower * upper_elem[j + 35];
					result_elem[j + 36] += lower * upper_elem[j + 36];
					result_elem[j + 37] += lower * upper_elem[j + 37];
					result_elem[j + 38] += lower * upper_elem[j + 38];
					result_elem[j + 39] += lower * upper_elem[j + 39];
				}
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
		register double *m1_elem = &m1[i * N];
		register double *m2_elem = &m2[i * N];
		register double *res = &result[i * N];

		for (j = 0; j < N; j++) {
			res[j] = m1_elem[j] + m2_elem[j];
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
