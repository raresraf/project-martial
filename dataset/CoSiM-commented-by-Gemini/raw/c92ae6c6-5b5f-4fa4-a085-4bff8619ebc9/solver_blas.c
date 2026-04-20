/**
 * @file solver_blas.c
 * @brief BLAS-based optimized matrix solver computing $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Relies on optimized numerical linear algebra kernels (DGEMM, DTRMM).
 * Time Complexity: $O(N^3)$ due to dense matrix multiplications.
 * Space Complexity: $O(N^2)$ for storing the result matrix C.
 */



#include "utils.h"
#include <cblas.h>




/**
 * @brief Computes C = A * B * B^T + A^T * A.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
double* my_solver(int N, double *A, double *B) {

printf("BLAS SOLVER\n");


double *C=(double *)calloc(N*N,sizeof(double));
double *B2=(double *)calloc(N*N,sizeof(double));
double *A2=(double *)calloc(N*N,sizeof(double));


cblas_dcopy(N*N,B,1,B2,1);
cblas_dcopy(N*N,A,1,A2,1);



cblas_dtrmm(CblasRowMajor,     
			CblasLeft,         
            CblasUpper,        
			CblasNoTrans,      
            CblasNonUnit,      
			N,                 
			N,                 
            1,                 
			A,                 
			N,                 
            B,                 
			N);                



cblas_dgemm(CblasRowMajor,     
 			CblasNoTrans,      
            CblasTrans,        
			N,                 
			N,                 
            N,                 
			1,                 
			B,                 
            N,                 
			B2,                
			N,                 
            0,                 
			C,                 
			N                  
			);




cblas_dtrmm(CblasRowMajor,     
			CblasLeft,         
            CblasUpper,        
			CblasTrans,        
            CblasNonUnit,      
			N,                 
			N,                 
            1,                 
			A2,                
			N,                 
            A,                 
			N                  
			);



cblas_daxpy	(N*N,              
			1,                 
 			A,                 
            1,                 
			C,                 
			1                  
			);


free(B2);
free(A2);

return (C);

	
}
