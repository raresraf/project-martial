
/**
 * @file compare.c
 * @brief This file provides utility functions for comparing two matrix files for equality
 * within a specified floating-point precision. It uses memory-mapped files for efficient
 * access to large datasets.
 *
 * Algorithm: Memory-mapped file comparison with element-wise floating-point precision check.
 * Time Complexity: O(N^2) where N is the dimension of the square matrix (or sqrt of total elements).
 * Space Complexity: O(1) additional space (beyond memory-mapped files), as it compares elements in-place.
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <math.h>

/**
 * @brief Utility macro to compare two double-precision floating-point numbers within a given error tolerance.
 * Functional Utility: Facilitates fuzzy comparison of floating-point values to account for potential
 * precision differences inherent in floating-point arithmetic.
 *
 * @param a The first floating-point number.
 * @param b The second floating-point number.
 * @param err The maximum allowed absolute difference for `a` and `b` to be considered equal.
 * @return 0 if the absolute difference between `a` and `b` is less than or equal to `err`, otherwise -1.
 */
#define check_err(a,b,err) ((fabs((a) - (b)) <= (err)) ? 0 : -1)


/**
 * @brief Compares two matrix files for equality within a specified floating-point precision.
 * This function memory-maps both input files for efficient read access and performs an
 * element-wise comparison. If a difference exceeding the given precision is found,
 * it reports the discrepancy and terminates the comparison.
 *
 * @param file_path1 The path to the first matrix file.
 * @param file_path2 The path to the second matrix file.
 * @param precision The maximum allowed difference between corresponding elements for them to be considered equal.
 * @return 0 if the files are identical within the specified precision, -1 otherwise.
 */
int cmp_files(char const *file_path1, char const *file_path2, double precision) {
	struct stat fileInfo1, fileInfo2;
	double *mat1, *mat2;
	int i, j, fd1, fd2, N, ret = 0;

	// Block Logic: Open both matrix files for read-only access.
	// Precondition: `file_path1` and `file_path2` are valid and accessible file paths.
	// Invariant: `fd1` and `fd2` hold valid file descriptors if successful.
	fd1 = open(file_path1, O_RDONLY, (mode_t)0600);
	fd2 = open(file_path2, O_RDONLY, (mode_t)0600);

	// Functional Utility: Retrieve file status information, including file size.
	fstat(fd1, &fileInfo1);
	fstat(fd2, &fileInfo2);

	// Block Logic: Check if the file sizes differ.
	// Precondition: `fileInfo1` and `fileInfo2` contain valid file statistics.
	// Invariant: If sizes differ, an error message is printed, file descriptors are closed, and -1 is returned.
	if(fileInfo1.st_size != fileInfo2.st_size) {
		printf("Files length differ\n");
		close(fd1);
		close(fd2);
		return -1;
	}

	// Block Logic: Memory-map the first file into virtual address space.
	// Precondition: `fd1` is a valid file descriptor and `fileInfo1.st_size` is accurate.
	// Invariant: `mat1` points to the memory-mapped content, or an error is handled.
	mat1 = (double*) mmap(0, fileInfo1.st_size, PROT_READ, MAP_SHARED, fd1, 0);
	if (mat1 == MAP_FAILED)
	{
		close(fd1);
		close(fd2);
		printf("Error mmapping the first file");
		return -1;
	}

	// Block Logic: Memory-map the second file into virtual address space.
	// Precondition: `fd2` is a valid file descriptor and `fileInfo2.st_size` is accurate.
	// Invariant: `mat2` points to the memory-mapped content, or an error is handled (unmapping `mat1` first).
	mat2 = (double*) mmap(0, fileInfo2.st_size, PROT_READ, MAP_SHARED, fd2, 0);
	if (mat2 == MAP_FAILED)
	{
		munmap(mat1, fileInfo1.st_size);
		close(fd1);
		close(fd2);
		printf("Error mmapping the second file");
		return -1;
	}

	// Functional Utility: Calculate the dimension N of the square matrices.
	// Invariant: `N` is derived from the total size of the file and the size of a double.
	N = sqrt(fileInfo1.st_size / sizeof(double));

	// Block Logic: Perform element-wise comparison of the two matrices.
	// Precondition: `mat1`, `mat2` are valid memory-mapped pointers, `N` is the matrix dimension.
	// Invariant: Loop iterates through all elements; if a difference exceeds `precision`, the loop breaks.
	for (i = 0; i < N; i++ ) {
		for (j = 0; j< N; j++) {
			// Functional Utility: Check if the absolute difference between elements is within the specified precision.
			ret = check_err(mat1[i * N + j], mat2[i * N + j], precision); 
			if (ret != 0) {
				printf("Matrixes differ on index [%d, %d]. Expected %.8lf got %.8lf\n",
						i, j, mat1[i * N + j], mat2[i * N + j]);
				goto done;
			}
		}
	}

done:
	// Functional Utility: Unmap memory-mapped files to release resources.
	munmap(mat1, fileInfo1.st_size);
	munmap(mat2, fileInfo2.st_size);

	// Functional Utility: Close file descriptors.
	close(fd1);
	close(fd2);

	return ret;
}

/**
 * @brief Main entry point of the comparison utility.
 * Parses command-line arguments for two matrix file paths and a precision tolerance,
 * then invokes `cmp_files` to perform the comparison. It prints the result to standard output.
 *
 * @param argc The number of command-line arguments.
 * @param argv An array of command-line argument strings. Expected format: `[program_name] [matrix_file1] [matrix_file2] [tolerance]`.
 * @return 0 if the matrices are identical within the given tolerance, -1 otherwise.
 */
int main(int argc, const char **argv)
{    
	double precision;
	int ret = 0;

	// Block Logic: Validate the number of command-line arguments.
	// Precondition: `argc` is the count of arguments provided to the program.
	// Invariant: If `argc` is less than 4, print usage instructions and exit with an error code.
	if(argc < 4) {
		printf("Usage: %s mat1 mat2 tolerance\n",argv[0]);
		exit(-1);
	}

	// Functional Utility: Parse the precision tolerance from the third command-line argument.
	sscanf(argv[3], "%lf", &precision);

	// Functional Utility: Call `cmp_files` to perform the matrix comparison.
	ret = cmp_files(argv[1],argv[2],precision);
	
	// Block Logic: Print the comparison result.
	// Precondition: `ret` contains the result of `cmp_files`.
	// Invariant: A message indicating "OK" or "Incorrect results!" is printed along with the program arguments.
	printf("%s %s %s %s\n", argv[0], argv[1], argv[2], (ret == 0 ? "OK" : "Incorrect results!"));
	return ret;
}
