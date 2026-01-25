
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	double *C;
	double *AtA, *BBt, *ABBt;
	double *At, *Bt;
	register int i;
	register int j;
	register int k;

	register int size = N * N * sizeof(*C);
	C = malloc(size);
	AtA = malloc(size);
	At = malloc(size);
	Bt = malloc(size);
	BBt = malloc(size);
	ABBt = malloc(size);
	if (C == NULL || AtA == NULL || BBt == NULL || ABBt == NULL) {
		exit(EXIT_FAILURE);
	}

	for (i = 0; i < N; ++i) {
		register double *At_col = At + i;
		register double *Bt_col = Bt + i;
		register double *pA = A + i * N;
		register double *pB = B + i * N;

		for (j = 0; j < N; ++j) {
			*At_col = *pA;
			*Bt_col = *pB;
			At_col += N;
			Bt_col += N;
			++pA;
			++pB;
		}
	}

	
	register double *AtA_ptr = AtA;
	for (i = 0; i < N; ++i) {
		register double *orig_pat = At + i * N;
		for (j = 0; j < N; ++j) {
			register double *pat = orig_pat;
			register double *pa = A + j;
			register double suma = 0.0;
			for (k = 0; k <= i; ++k) {
				suma += *pat * *pa;
				++pat;
				pa += N;
			}
			*AtA_ptr = suma;
			++AtA_ptr;
		}
	}

	
	register double *BBt_ptr = BBt;
	for (i = 0; i < N; ++i) {
		register double *orig_pb = B + i * N;
		for (j = 0; j < N; ++j) {
			register double *pb = orig_pb;
			register double *pbt = Bt + j;
			register double suma = 0.0;
			for (k = 0; k < N; ++k) {
				suma += *pb * *pbt;
				++pb;
				pbt += N;
			}
			*BBt_ptr = suma;
			++BBt_ptr;
		}
	}

	
	register double *ABBt_ptr = ABBt;
	for (i = 0; i < N; ++i) {
		register double *orig_pa = A + i * N;
		for (j = 0; j < N; ++j) {
			register double *pa = orig_pa + i;
			register double *pbbt = BBt + i * N + j;
			register double suma = 0.0;
			for (k = i; k < N; ++k) {
				suma += *pa * *pbbt;
				++pa;
				pbbt += N;
			}
			*ABBt_ptr = suma;
			++ABBt_ptr;
		}
	}

	
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			*(C + i * N + j) = *(ABBt + i * N + j) + *(AtA + i * N + j);
		}
	}

	free(ABBt);
	free(AtA);
	free(BBt);
	free(At);
	free(Bt);
	return C;	
}
