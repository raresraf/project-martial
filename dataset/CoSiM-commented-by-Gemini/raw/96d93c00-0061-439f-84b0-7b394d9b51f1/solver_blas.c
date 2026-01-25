
#include "utils.h"
#include "cblas.h"
#include <string.h>


double* my_solver(int N, double *A, double *B) {
	double *A_star_B;
	double *OP1;
	double *OP2;
	double *RES;
	int numMatrixElems = N * N;

	A_star_B = calloc(numMatrixElems, sizeof(*A_star_B));
	OP1 = calloc(numMatrixElems, sizeof(*OP1));
	OP2 = calloc(numMatrixElems, sizeof(*OP2));
	RES = calloc(numMatrixElems, sizeof(*RES));

	memcpy(A_star_B, B, numMatrixElems * sizeof(*B));

	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1.0, A, N, A_star_B, N);

	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1.0, A, N, A, N);

	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, A_star_B, N, B, N, 1.0, A, N);

	memcpy(RES, A, numMatrixElems * sizeof(*A));
	free(A_star_B);
	free(OP1);
	free(OP2);
	return RES;
}
