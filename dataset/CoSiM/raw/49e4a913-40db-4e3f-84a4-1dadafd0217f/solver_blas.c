
#include "utils.h"
#include <string.h>
#include <stdlib.h>
#include "cblas.h"

double* my_solver(int N, double *A, double *B) {
	
	double * C, *AB;

	
	C = calloc(N * N, sizeof(double));
	if(C == NULL)
		printf("Probleme la alocarea memoriei\n");
	AB = calloc(N * N, sizeof(double));
	if(AB == NULL)
		printf("Probleme la alocarea memoriei\n");
	memcpy(C, A, N * N * sizeof(*C));
	cblas_dtrmm(CblasRowMajor,
		CblasLeft, CblasUpper, CblasTrans, CblasNonUnit,
		N, N,
		1.0, A, N,
		C, N   );
	memcpy(AB, B, N * N * sizeof(*AB));
	cblas_dtrmm(CblasRowMajor,
                CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                N, N,
                1.0, A, N,
                AB, N   );
	cblas_dgemm(CblasRowMajor,
                CblasNoTrans,
		CblasTrans,
		N, N, N, 1.0,
		AB, N, B, N,
		1.0, C, N  
	);
	free(AB);

	return C;
}
