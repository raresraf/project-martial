
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");

	register int i, j, k;
	register double aux;
	
	double *BBt = (double*) calloc(N * N, sizeof(double));

	for (i = 0; i < N; ++i) {
		register double *orig_pb = &B[i * N + 0];
		for (j = 0; j < N; ++j) {
			register double *pb = orig_pb;
			register double *pb_t = &B[j * N + 0];
			aux = 0.0;

			for (k = 0; k < N; ++k) {
				aux += *pb * *pb_t;
				++pb;
				++pb_t;
			}

			BBt[i * N + j] = aux;
		}
	}

	double *ABBt = (double*) calloc(N * N, sizeof(double));

	for (i = 0; i < N; ++i) {
		register double *orig_pa = &A[i * N + i];
		for (j = 0; j < N; ++j) {
			register double *pa = orig_pa;
			register double *pbbt = &BBt[i * N + j];
			aux = 0.0;

			for (k = i; k < N; ++k) {
				aux += *pa * *pbbt;
				++pa;
				pbbt += N;
			}

			ABBt[i * N + j] = aux;
		}
	}

	free(BBt);

	double *AAt = (double*) calloc(N * N, sizeof(double));

	for (i = 0; i < N; ++i) {
		register double *orig_pa = &A[i];
		for (j = 0; j < N; ++j) {
			register double *pa = orig_pa;
			register double *pa_t = &A[j];
			aux = 0.0;

			for (k = 0; k <= ((i < j) ? i : j); ++k) {
				aux += *pa * *pa_t;
				pa += N;
				pa_t += N;
			}

			AAt[i * N + j] = aux;
		}
	}

	double *res = (double*) calloc(N * N, sizeof(double));

	for (i = 0; i < N * N; ++i) {
		res[i] = ABBt[i] + AAt[i];
	}

	free(ABBt);
	free(AAt);

	return res;
}
