
#include "utils.h"

#include <time.h>
#define SECOND_MICROS		1000000.f



void compare (int N, double *firstMatrix, double *secondMatrix) {
	int rowCounter, colCounter;
	int isEqual = 1;

	for(rowCounter=0; rowCounter<N && isEqual==1; rowCounter++){
        for(colCounter = 0; colCounter < N; colCounter++){
            if(firstMatrix[colCounter + N * rowCounter] != 
                    secondMatrix[colCounter + N * rowCounter]){
                printf("UNEQUAL MATRICES: Element mismatch\n");
                isEqual = 0;
                break;
            }
        }
    }
    if(isEqual == 1){
        printf("EQUAL MATRICES\n");
    }

}

double* my_solver(int N, double *A, double* B) {
	int i,j,k;
	double *C, *D;
	C = (double *)malloc(N * N * sizeof(double));
	D = (double *)malloc(N * N * sizeof(double));

	
	for (i = 0; i < N; i++) {
		double *orig_pa = &B[i * N];
		for (j = i; j < N; j++) {
			double *pa = orig_pa;
    		double *pb = &B[j * N];
			register double sum = 0.0;
			for (k = 0; k < N; k ++) {
				sum += *pa * *pb;
				pa++;
				pb++;
			}

			C[i + N * j] = C[j + N * i] = sum; 
		}	
	}

	
	for (i = 0; i < N; i++) {
		double *orig_pa = &A[i + i * N];
		for (j = 0; j < N; j++) {
			double *pa = orig_pa;
    		double *pb = &C[N * i + j];
			register double sum = 0.0;
			for (k = i; k < N; k ++) {
				sum += *pa * *pb;
				pa++;
				pb += N;
			}

			D[j + N * i] = sum;
		}	
	}

	free(C);
	C = (double *)malloc(N * N * sizeof(double));
	
	
	for (i = 0; i < N; i++) {
		double *orig_pa = &A[i];
		for (j = i; j < N; j++) {
			double *pa = orig_pa;
    		double *pb = &A[j];
			register double sum = 0.0;
			for (k = 0; k < i + 1; k++) {
				sum += *pa * *pb;
				pa += N;
				pb += N;
			}

			
			D[j + N * i] += sum;
			D[i + N * j] += sum;
		}	
	}

	free(C);
	
	

	
	
	

	

	
	
	
	
	
	
	
	

	
	
	

	
	
	
	
	
	
	
	
	
	

	
	

	
	
	
	
	
	
	
	

	
	
	
	
	
	

	
	
	
	
	
	
	

	


	return D;	
}
