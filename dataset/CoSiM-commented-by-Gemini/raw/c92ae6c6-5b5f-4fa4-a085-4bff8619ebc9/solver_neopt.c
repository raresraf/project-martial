/**
 * @file solver_neopt.c
 * @brief Unoptimized standard implementation to compute $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Naive multi-loop dense matrix multiplication with upper triangular conditions.
 * Time Complexity: $O(N^3)$.
 * Space Complexity: $O(N^2)$ to store the intermediate matrices.
 */



#include "utils.h"
#include <math.h>




int min(int a,int b){

/**
 * Block Logic: Conditional state branch.
 * Invariant: The conditional branch maintains control flow invariants.
 */
if(a>b)
	return b;
else
	return a;

}



void mul1 (double *A,double *B,double *C,int N){

int i,j,k;

/**
 * Block Logic: Iterative processing loop.
 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
 */
for(i=0;i<N;i++)
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for(j=0;j<N;j++)
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for(k=i;k<N;k++)  
			C[i*N+j]+=A[i*N+k]*B[k*N+j];

}



void mul2 (double *A,double *B,double *C,int N){

int i,j,k;

/**
 * Block Logic: Iterative processing loop.
 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
 */
for(i=0;i<N;i++)  
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for(j=0;j<N;j++)
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for(k=0;k<N;k++)
			C[i*N+j]+=A[i*N+k]*B[j*N+k];

}



void mul3 (double *A,double *C,int N){

int i,j,k;

/**
 * Block Logic: Iterative processing loop.
 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
 */
for(i=0;i<N;i++)
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for(j=0;j<N;j++)
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for(k=0;k<min(i,j)+1;k++) 
			C[i*N+j]+=A[k*N+j]*A[k*N+i];

}



void add(double *A,double *B,int N){

int i;

/**
 * Block Logic: Iterative processing loop.
 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
 */
for(i=0;i<N*N;i++)
		A[i]+=B[i];

}


/**
 * @brief Computes C = A * B * B^T + A^T * A.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
double* my_solver(int N, double *A, double* B) {

printf("NEOPT SOLVER\n");


double *C1=(double *)calloc(N*N,sizeof(double));
double *C2=(double *)calloc(N*N,sizeof(double));
double *C3=(double *)calloc(N*N,sizeof(double));


mul1(A,B,C1,N);

mul2(C1,B,C2,N);

mul3(A,C3,N);

add(C2,C3,N);


free(C1);
free(C3);

return C2;
}

