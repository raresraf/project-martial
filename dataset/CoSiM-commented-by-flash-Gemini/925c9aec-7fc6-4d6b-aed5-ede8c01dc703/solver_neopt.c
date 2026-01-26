
#include "utils.h"

double* my_solver(int N, double *A, double* B) {
	double *BBT, *ABBT, *ATA, *C, *AT;
	
	BBT = calloc(N * N , sizeof(double));
	ABBT = calloc(N * N , sizeof(double));
	ATA = calloc(N * N , sizeof(double));
	AT = calloc(N * N, sizeof(double));
	C = calloc(N * N, sizeof(double));

	
	
    for (int i = 0; i < N; ++i) {
      for (int j = i; j < N; ++j) {
         for (int k = 0; k < N; ++k) {
            BBT[j * N + i] = BBT[i * N + j] += B[i * N + k] * B[j * N + k];
         }
      }
   }

	
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
         for (int k = i; k < N; ++k) {
            ABBT[i * N + j] += A[i * N + k] * BBT[k * N + j];
         }
      }
   }

	
    for (register int i = 0; i < N; i++) {
        for (register int j = 0; j <= i ; j++) {
            AT[i * N + j] = A[j * N + i];
		}
	}
   
    
	
    for (int i = 0; i < N; ++i) {
		for (int j = i; j < N; ++j) {
			for (int k = 0; k < N; ++k) {
				ATA[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
   }

	
    for (register int i = 0; i < N; ++i) {
    	for (register int j = i; j < N; ++j) {
			C[i * N + j] = ABBT[i * N + j] + ATA[i * N + j];
			C[j * N + i] = ABBT[j * N + i] + ATA[i * N + j];
		}
	}

	free(BBT);
	free(ABBT);
	free(AT);
	free(ATA);

	return C;
}
