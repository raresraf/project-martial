
/**
 * @file utils.h
 * @brief Utility functions and data structures for a matrix solver test framework.
 *
 * This header defines a standardized `Solver` function pointer type, a macro for
 * generating random double-precision numbers, and a `test` struct for holding
 * test case parameters. It also declares function prototypes for running tests,
 * generating and freeing test data, and reading input files.
 */
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

/**
 * @typedef Solver
 * @brief A function pointer type for matrix solver functions.
 *
 * A `Solver` function takes the matrix dimension `N` and two `N x N` matrices
 * `A` and `B` as input, and returns a pointer to the resulting `N x N` matrix.
 */
typedef double* (*Solver)(int, double *, double*);

/**
 * @def get_rand_double(limit)
 * @brief Generates a random double-precision number within a given range.
 * @param limit The upper bound of the range [-limit, limit].
 * @return A random double-precision number.
 */
#define get_rand_double(limit) ((((double)rand()) / RAND_MAX) * (2 * limit) - limit)

/**
 * @struct test
 * @brief A struct to hold parameters for a single test case.
 */
struct test {
	int seed;
	int N;
	
	char output_save_file[100];
};

/**
 * @brief A placeholder for the actual solver implementation.
 *
 * This function is expected to be defined in a separate source file that
 * implements a specific matrix solving algorithm.
 *
 * @param N The dimension of the matrices.
 * @param A A pointer to the first input matrix.
 * @param B A pointer to the second input matrix.
 * @return A pointer to the resulting matrix.
 */
double* my_solver(int, double *, double *);

/**
 * @brief Runs a single test case.
 *
 * @param t The test case parameters.
 * @param solver A function pointer to the solver to be tested.
 * @param timing A pointer to a float where the execution time will be stored.
 * @return 0 on success, a non-zero value on failure.
 */
int run_test(struct test, Solver, float *);

/**
 * @brief Frees the memory allocated for the matrices.
 * @param data A pointer to an array of matrix pointers.
 */
void free_data(double **);

/**
 * @brief Generates random matrix data for a test case.
 *
 * @param t The test case parameters.
 * @param data A pointer to an array where the generated matrices will be stored.
 * @param num_matrices The number of matrices to generate.
 * @return 0 on success, a non-zero value on failure.
 */
int generate_data(struct test, double **, int);

/**
 * @brief Reads test case parameters from an input file.
 *
 * @param file_path The path to the input file.
 * @param num_tests A pointer to an integer where the number of tests will be stored.
 * @param tests A pointer to an array where the read test cases will be stored.
 * @return 0 on success, a non-zero value on failure.
 */
int read_input_file(char *, int *, struct test **);
