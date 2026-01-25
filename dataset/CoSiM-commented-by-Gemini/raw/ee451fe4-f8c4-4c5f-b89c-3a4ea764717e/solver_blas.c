
#include "utils.h"
#include "cblas.h"


double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");

	int i, j, idx;

	double * m1 = (double *)calloc(N * N, sizeof(double));
	if (m1 == NULL) 
		return NULL;

	double * m2 = (double *)calloc(N * N, sizeof(double));
	if (m2 == NULL) 
		return NULL;

	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			idx = i * N + j;

			m1[idx] = B[idx];
			m2[idx] = A[idx];
		}
	}

	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
				N, N, 1, A, N, m1, N);

	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit,
				N, N, 1, A, N, m2, N);

	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
				N, N, N, 1, m1, N, B, N, 1, m2, N);


	free(m1);

	return m2;
}
