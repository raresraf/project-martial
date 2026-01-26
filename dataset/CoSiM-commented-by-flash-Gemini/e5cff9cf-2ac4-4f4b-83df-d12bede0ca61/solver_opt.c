
#include "utils.h"

double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
	double *C = calloc(N * N, sizeof(double));
	if (!C)
		return NULL;

	double *D = calloc(N * N, sizeof(double));
	if (!D)
		return NULL;

	double *Bt = malloc(N * N * sizeof(double));
	if (!Bt)
		return NULL;

	int i, j, k;
	register double cnst;
	register double *point_a, *point_b, *point_bt, *point_c, *point_d;

	for (i = 0; i < N; ++i) {
		point_b = B + i * N - 1;
		point_bt = Bt + i;
		for (j = 0; j < N; ++j) {
			*(point_bt) = *(++point_b);
			point_bt += N;
		}
	}

	for (i = 0; i < N; ++i) {
		for (k = 0; k < N; ++k) {
			cnst = B[i * N + k];
			point_c = C + i * N + i- 1;
			point_bt = Bt + k * N + i - 1;
			for (j = i; j < N; ++j) {
				*(++point_c) += cnst * *(++point_bt);
			}
		}
	}

	for (i = 0; i < N; ++i) {
		for (k = i; k < N; ++k) {
			cnst = A[i * N + k];
			point_d = D + i * N - 1;
			point_c = C + k;
			
			for (j = 0; j < k; ++j) {
				*(++point_d) += cnst * *(point_c);
				point_c += N;
			}
			--point_c;
			for (j = k; j < N; ++j) {
				*(++point_d) += cnst * *(++point_c);
			}
		}
	}

	for (i = 0; i < N; ++i) {
		for (k = 0; k <= i; ++k) {
			cnst = A[k * N + i];
			point_d = D + i * N + k - 1;
			point_a = A + k * N + k - 1;
			for (j = k; j < N; ++j) {
				*(++point_d) += cnst * *(++point_a);
			}
		}
	}

	free(C);
	free(Bt);
	return D;	
}
