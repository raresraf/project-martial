
#include <stdlib.h>
#include "utils.h"

double *my_solver(int N, double *A, double *B) {

        register int i, j, k;

        double *At = calloc(N * N, sizeof(double));
        double *B2 = calloc(N * N, sizeof(double));
        double *B3 = calloc(N * N, sizeof(double));

        if (At == NULL || B2 == NULL || B3 == NULL) {
                return NULL;
        }


        
        for (i = 0; i != N; ++i) {

                register double *ptr_At = At + i;

                register double *ptr_A = A + i * N;

                for (j = 0; j != N; ++j) {
                       *ptr_At = *ptr_A;
                        ++ptr_A;
                        ptr_At += N;
                }
        }


        
        for (i = 0; i != N; ++i) {

                
                register double *ptrB2 = B2 + i * N;

                
                register double *B_aux = B + i * N;

                
                for (j = 0; j != N; ++j) {
                        
                        register double sum = 0.0;

                        
                        register double *ptrA = A + i * N + i;

                        
                        register double *ptrB = B_aux + j;

                        
                        for (k = i; k != N; ++k) {
                                sum += *ptrA * *ptrB;
                                ++ptrA;
                                ptrB += N;
                        }

                        *ptrB2 = sum;
                        ++ptrB2;
                }
        }


        
        for (i = 0; i != N; ++i) {

                
                register double *ptrB3 = B3 + i * N;

                for (j = 0; j != N; ++j) {
                        
                        register double sum = 0.0;

                        
                        register double *ptrB2 = B2 + i * N;

                         
                        register double *ptrB = B + j * N;

                        for (k = 0; k != N; ++k) {
                                sum += *ptrB2 * *ptrB;
                                ++ptrB2;
                                ++ptrB;
                        }

                        *ptrB3 = sum;
                        ++ptrB3;
                }
        }

        
        for (i = 0; i != N; ++i) {
    
                
                register double *aux_B3 = B3;

                
                register double *ptr_At = At + i;

                for (j = 0; j != N; ++j) {

                        
                        register double *ptr_B3 = aux_B3;

                        
                        register double *ptrA = A + i * N;

                        for (k = 0; k != N; ++k) {
                            *ptr_B3 += *ptr_At * *ptrA;
                            ++ptr_B3;
                            ++ptrA;
                        }

                        ptr_At += N;
                        aux_B3 += N;
                }
        }
        

        free(At);
        free(B2);

	return B3;
}
