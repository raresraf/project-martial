
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");

    
	double *BBt = (double *)calloc(N * N, sizeof(double));
	for (int i = 0; i < N; i += 2) {
        
	    register int row = i * N;
        register double *original_b = &(B[row]);
        register double *bbt = &(BBt[row]);

		for (int j = 0; j < N; j += 2) {
            
            register double sum1 = 0.0, sum2 = 0.0, sum3 = 0.0, sum4 = 0.0;

            
            register double *b1 = original_b;
            register double *b2 = &(B[j * N]);

            
			for (int k = 0; k < N; k++) {
				sum1 += (*b1) * (*b2);
                sum2 += (*(b1 + N)) * (*b2);
                sum3 += (*b1) * (*(b2 + N));
                sum4 += (*(b1 + N)) * (*(b2 + N));
                b1++;
                b2++;
			}

            
            *(bbt) = sum1;
            *(bbt + N) = sum2;
            *(bbt + 1) = sum3;
            *(bbt + N + 1) = sum4;
            bbt += 2;
		}
	}

    
    double *A_BBt = (double *)calloc(N * N, sizeof(double));
    for (int i = 0; i < N; i += 2) {
        
        register int row = i * N;
        register double *a = &(A[row + i]);

        
		for (int k = i; k < N; k++) {
            
            register double *bbt = &(BBt[k * N]);
            register double *a_bbt = &(A_BBt[row]);

            
			for (int j = 0; j < N; j += 2) {
                *(a_bbt) += (*a) * (*bbt);
				*(a_bbt + N) += (*(a + N)) * (*bbt);
                *(a_bbt + 1) += (*a) * (*(bbt + 1));
                *(a_bbt + N + 1) += (*(a + N)) * (*(bbt + 1));
				bbt += 2;
                a_bbt += 2;
			}
			a++;
		}
	}

    
    double *AtA = (double *)calloc(N * N, sizeof(double));
    for (int i = 0; i < N; i++) {
        
		register int row =  i * N;

        
		for (int k = 0; k <= i; k++) {
            
            register double *a2 = &(A[k * N]);
            register double *a1 = a2 + i;
            register double *ata = &(AtA[row]);

			for (int j = 0; j < N; j++) {
				*(ata) += (*a1) * (*a2);
                a2++;
                ata++;
			}
		}
	}

    
    for (int i = 0; i < N; i += 2) {
        
        register int row = i * N;
        register double *ata = &(AtA[row]);
        register double *a_bbt = &(A_BBt[row]);

        
		for (int j = 0; j < N; j += 2) {
			*(a_bbt) += (*ata);
            *(a_bbt + N) += (*(ata + N));
            *(a_bbt + 1) += (*(ata + 1));
            *(a_bbt + N + 1) += (*(ata + N + 1));
            ata += 2;
            a_bbt += 2;
		}
	}

    
    free(AtA);
    free(BBt);
	return A_BBt;
}
