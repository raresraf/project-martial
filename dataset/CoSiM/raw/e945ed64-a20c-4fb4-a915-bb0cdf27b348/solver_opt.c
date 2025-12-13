
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	register int i, j, k;

	double *C = calloc(N * N, sizeof(*C));
	if (C == NULL) {
		exit(EXIT_FAILURE);
	}

	double *A_tA = calloc(N * N, sizeof(*A_tA));
	if (A_tA == NULL) {
		exit(EXIT_FAILURE);
	}

	double *AB = calloc(N * N, sizeof(*AB));
	if (AB == NULL)
		exit(EXIT_FAILURE);

	double *ABB_t = calloc(N * N, sizeof(*ABB_t));
	if (ABB_t == NULL)
		exit(EXIT_FAILURE);

	
	for (i = 0; i < N; ++i) {
		register double *orig_pa = A + i * N;
		for (j = 0; j < N; ++j) {
			register double *pa = orig_pa;
			register double *pb = B + j;
			register double sum = 0.0;
			for (k = 0; k < N; ++k) {
				if (k >= i) {
					sum += *pa * *pb;
				}
				
				pa++;
				pb += N;
			}

			AB[i * N + j] = sum;
		}
	}

	
	for (i = 0; i < N; ++i) {
		register double *orig_pab = AB + i * N;
		for (j = 0; j < N; ++j) {
			register double *pab = orig_pab;
			register double *pb_t = B + j * N;
			register double sum = 0.0;
			for (k = 0; k < N; ++k) {
				sum += *pab * *pb_t;
				pab++;
				pb_t++;
			}

			ABB_t[i * N + j] = sum;
		}
	}

	
	for (i = 0; i < N; ++i) {
		register double *orig_pa_t = A + i;
		for (j = 0; j < N; ++j) {
			register double *pa_t = orig_pa_t;
			register double *pa = A + j;
			register double sum = 0.0;
			for (k = 0; k <= i; ++k) {
				sum += *pa_t * *pa;
				pa_t += N;
				pa += N;
			}

			A_tA[i * N + j] = sum;
		}
	}

	
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			C[i * N + j] = ABB_t[i * N + j] + A_tA[i * N + j];
		}
	}

	free(A_tA);
	free(AB);
	free(ABB_t);

	return C;
}
