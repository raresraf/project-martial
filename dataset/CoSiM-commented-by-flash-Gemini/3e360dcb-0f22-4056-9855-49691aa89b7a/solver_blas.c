
#include "utils.h"
#include "cblas.h"
#include <string.h>


double* my_solver(int N, double *A, double *B) {
	double *C;
	double *AtA, *BBt, *ABBt;
	int i, j;

	C = calloc(N * N, sizeof(*C));
	AtA = calloc(N * N, sizeof(*AtA));
	BBt = calloc(N * N, sizeof(*BBt));
	ABBt = calloc(N * N, sizeof(*ABBt));

	
	memcpy(AtA, A, N * N * sizeof(*C));
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1.0, A, N, AtA, N);

	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, B, N, B, N, 1.0, BBt, N);

	
	memcpy(ABBt, BBt, N * N * sizeof(*ABBt));
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1.0, A, N, ABBt, N);

	
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			C[i * N + j] += AtA[i * N + j] + ABBt[i * N + j];
		}
	}
	
	free(AtA);
	free(BBt);
	free(ABBt);
	return C;
}
