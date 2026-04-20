/**
 * @file solver_opt.c
 * @brief Block-optimized manual implementation to compute $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Loop tiling/blocking for cache locality enhancement.
 * Time Complexity: $O(N^3)$.
 * Space Complexity: $O(N^2)$ for intermediate data structures.
 */



#include "utils.h"




void mul1 (double *A,double *B,double *C,int N){

int i,j,k;
double *pa,*pb,*pc; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
double *orig_pc;


/**
 * Block Logic: Iterative processing loop.
 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
 */
for(i = 0; i < N; i++){
	orig_pc = &C[i*N]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
	pa=&A[i*N+i];
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for(k = i; k < N; k++){
		pc=orig_pc;
		register double cconst = *pa;  /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		pa++;
		pb=&B[k*N];
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for(j = 0; j < N; j++){
			*pc += cconst * *pb;
			pc++;
			pb++;
			
			}
		}
	}

}


void mul2 (double *A,double *B,double *C,int N){

int i,j,k;
double *pa,*pb,*pc; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
double *orig_pa;


pc=&C[0];
/**
 * Block Logic: Iterative processing loop.
 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
 */
for(i = 0; i < N; i++){
	orig_pa = &A[i*N]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
	pb=&B[0];
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for(j = 0; j < N; j++){
		pa=orig_pa;
		register double cconst = 0;
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for(k = 0; k < N; k++){
			cconst += *pa * *pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			pa++;
			pb++;
			
			}
		*pc=cconst;
		pc++;
	    }
	}



}


void mul3 (double *A,double *C,int N){

int i,j,k;
double *pa,*pb,*pc; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
double *orig_pb;


/**
 * Block Logic: Iterative processing loop.
 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
 */
for(i=0;i<N;i++){
pa=&A[i*N+i];
orig_pb=&A[i*N+i];
pc=&C[i*N];
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for(j=i;j<N;j++){
		pb=orig_pb;
		register double cconst = *pa; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		pa++;
		pc+=i;
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for(k=i;k<N;k++){
			*pc += cconst * *pb;
      		pc++;
      		pb++;
			
			}
		}
	}


}


void add(double *A,double *B,int N){

int i;
double *pa,*pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
pa=&A[0];
pb=&B[0];

/**
 * Block Logic: Iterative processing loop.
 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
 */
for(i=0;i<N*N;i++){
	*pa+=*pb;
	pa++;
	pb++;
	}

}


/**
 * @brief Computes C = A * B * B^T + A^T * A.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
double* my_solver(int N, double *A, double* B) {

printf("OPT SOLVER\n");


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
