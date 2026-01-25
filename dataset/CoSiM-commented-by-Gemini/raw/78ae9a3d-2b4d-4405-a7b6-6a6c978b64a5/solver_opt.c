
#include "utils.h"


double* my_solver(int N, double *A, double *B) {
	printf("OPT SOLVER\n");
	register int i, j, k;
	double *aux1 = (double*) calloc(N * N, sizeof(double));
	double *aux3 = (double*) calloc(N * N, sizeof(double));
	double *A_tr = (double*) calloc(N * N, sizeof(double));
	double *B_tr = (double*) calloc(N * N, sizeof(double));

	for (i = 0; i < N; ++i) {
		register int count1 = N * i;
		register double *pa = &A[count1];
		register double *pb = &B[count1];
		register double *pa_tr = &A_tr[i];
		register double *pb_tr = &B_tr[i];

		for (j = 0; j < N; ++j) {
			*pa_tr = *pa;
			*pb_tr = *pb;
			pa++;
			pb++;
			pa_tr += N;
			pb_tr += N;
		}
	}

	for (i = 0; i < N; ++i) {
		register int count1 = N * i;
		register double *orig_pa = &A[count1];
		register double *orig_pa_tr = &A_tr[count1];
		for (j = 0; j < N; ++j) {
			register int count2 = N * i + j;
			register double *paux1 = &aux1[count2];
			register double *paux3 = &aux3[count2];

			register double *pa = orig_pa;
			register double *pa_tr = orig_pa_tr;
  			register double *pb = &B[j];
			register double *pa2 = &A[j];
			
			for (k = 0; k < N; ++k) {
				if (i <= k) {
					*paux1 += *pa * *pb;
				}
				if (k <= i && k <= j) {
					*paux3 += *pa_tr * *pa2;
				}
				pa++;
				pa_tr++;
				pb += N;
				pa2 += N;
			}
		}
	}

	for (i = 0; i < N; ++i) {
		register int count1 = N * i;
		register double *orig_pa = &aux1[count1];
		for (j = 0; j < N; ++j) {
			register int count2 = N * i + j;
			register double *paux3 = &aux3[count2];
			register double *pa = orig_pa;
  			register double *pb = &B_tr[j];
			for (k = 0; k < N; ++k) {
				*paux3 += *pa * *pb;
				pa++;
				pb += N;
			}
		}
	}
	free(aux1);
	free(A_tr);
	free(B_tr);
	return aux3;
}