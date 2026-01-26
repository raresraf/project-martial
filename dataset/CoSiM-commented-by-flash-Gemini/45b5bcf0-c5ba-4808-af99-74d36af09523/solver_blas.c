
#include "utils.h"
#include <string.h>
#include <cblas.h>


double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	double *first_mul = calloc (N * N, sizeof(double));

	if (!first_mul)
		return NULL;

	double *second_mul = calloc (N * N, sizeof(double));

	if (!second_mul)
		return NULL;

	double *third_mul = calloc (N * N, sizeof(double));

	if (!third_mul)
		return NULL;

	double *result = malloc (N * N * sizeof(double));

	if (!result)
		return NULL;

	
	memcpy(first_mul, B, N * N * sizeof(double));
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans,
		CblasNonUnit, N, N, 1.0, A, N, first_mul, N);

	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
		N, N, N, 1.0, first_mul, N, B, N, 1.0, second_mul, N);

	
	memcpy(third_mul, A, N * N * sizeof(double));
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans,
		CblasNonUnit, N, N, 1.0, A, N, third_mul, N);

	register int i, j;

	for (i = 0; i < N; i++) {
		register double *res = &result[i * N];
		register double *pa = &second_mul[i * N];
		register double *pb = &third_mul[i * N];

		for (j = 0; j < N; j++) {
			*res = *pa + *pb;
			res++;
			pa++;
			pb++;
		}
		res += N;
		pa += N;
		pb += N;
	}

	free(first_mul);
	free(second_mul);
	free(third_mul);

	return result;
}
