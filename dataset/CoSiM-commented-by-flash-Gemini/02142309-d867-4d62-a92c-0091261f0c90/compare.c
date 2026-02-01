
/**
 * @file compare.c
 * @brief Utility for comparing two binary files, assumed to contain square matrices of double-precision floating-point numbers.
 * It verifies if the files have identical sizes and then performs an element-wise comparison of their contents
 * within a specified floating-point precision tolerance. Designed for validating numerical computation results.
 * Algorithm: Memory-mapped file I/O for efficient data access, followed by nested loops for element-by-element comparison.
 * Time Complexity: $O(N^2)$, where N is the dimension of the square matrix.
 * Space Complexity: $O(1)$ additional space, as files are memory-mapped rather than loaded entirely into memory.
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <math.h>

#define check_err(a,b,err) ((fabs((a) - (b)) <= (err)) ? 0 : -1)

/**
 * @brief Compares two files that are assumed to contain matrices of doubles.
 * This function memory-maps both files for efficient access and performs an element-wise
 * comparison, returning 0 if they are identical within the given precision, and -1 otherwise.
 * @param file_path1 Path to the first file.
 * @param file_path2 Path to the second file.
 * @param precision The maximum allowed absolute difference between corresponding elements for them to be considered equal.
 * @return 0 if files are identical within precision, -1 otherwise.
 */
int cmp_files(char const *file_path1, char const *file_path2, double precision) {
	struct stat fileInfo1, fileInfo2;
	double *mat1, *mat2;
	int i, j, fd1, fd2, N, ret = 0;

	// Block Logic: Open both files in read-only mode.
	fd1 = open(file_path1, O_RDONLY, (mode_t)0600);
	fd2 = open(file_path2, O_RDONLY, (mode_t)0600);

	// Block Logic: Retrieve file status information, including size.
	fstat(fd1, &fileInfo1);
	fstat(fd2, &fileInfo2);

	// Conditional Logic: Check if file sizes differ.
	// Precondition: Both files must have the same size to represent matrices of the same dimensions.
	if(fileInfo1.st_size != fileInfo2.st_size) {
		printf("Files length differ\n");
		close(fd1);
		close(fd2);
		return -1;
	}

	// Block Logic: Memory-map the first file into virtual address space for direct access.
	// This avoids explicit read calls and provides efficient access to large files.
	mat1 = (double*) mmap(0, fileInfo1.st_size, PROT_READ, MAP_SHARED, fd1, 0);
	// Conditional Logic: Handle memory-mapping failure for the first file.
	if (mat1 == MAP_FAILED)
	{
		close(fd1);
		close(fd2);
		printf("Error mmapping the first file");
		return -1;
	}

	// Block Logic: Memory-map the second file into virtual address space.
	mat2 = (double*) mmap(0, fileInfo2.st_size, PROT_READ, MAP_SHARED, fd2, 0);
	// Conditional Logic: Handle memory-mapping failure for the second file.
	if (mat2 == MAP_FAILED)
	{
		munmap(mat1, fileInfo1.st_size); // Clean up first mmap.
		close(fd1);
		close(fd2);
		printf("Error mmapping the second file");
		return -1;
	}

	// Functional Utility: Calculate the dimension N of the square matrix.
	// Invariant: File size must be a multiple of sizeof(double) and a perfect square when divided by sizeof(double).
	N = sqrt(fileInfo1.st_size / sizeof(double));

	// Block Logic: Iterate through each element of the matrices for comparison.
	// Precondition: Both matrices are square and of dimension N.
	for (i = 0; i < N; i++ ) {
		for (j = 0; j< N; j++) {
			// Inline: Access matrix elements using row-major order: mat[row * N + col].
			ret = check_err(mat1[i * N + j], mat2[i * N + j], precision); 
			// Conditional Logic: If a difference is found beyond the specified precision, print details and exit.
			if (ret != 0) {
				printf("Matrixes differ on index [%d, %d]. Expected %.8lf got %.8lf\n",
						i, j, mat1[i * N + j], mat2[i * N + j]);
				// Inline: Jump to cleanup label upon first detected error.
				goto done;
			}
		}
	}

done:
	// Block Logic: Clean up resources by unmapping memory and closing file descriptors.
	munmap(mat1, fileInfo1.st_size);
	munmap(mat2, fileInfo2.st_size);

	close(fd1);
	close(fd2);

	return ret;
}

/**
 * @brief Main function for the matrix comparison utility.
 * Parses command-line arguments (two file paths and a precision tolerance)
 * and calls `cmp_files` to perform the comparison. Prints the result to stdout.
 * @param argc The number of command-line arguments.
 * @param argv An array of strings representing the command-line arguments.
 * @return 0 on successful comparison with identical results, -1 on error or differing results.
 */
int main(int argc, const char **argv)
{    
	double precision;
	int ret = 0;

	// Conditional Logic: Check for correct number of command-line arguments.
	// Precondition: Expects three arguments: mat1_path, mat2_path, and precision.
	if(argc < 4) {
		printf("Usage: %s mat1 mat2 tolerance\n",argv[0]);
		exit(-1);
	}

	// Functional Utility: Parse the precision tolerance from the third argument.
	sscanf(argv[3], "%lf", &precision);

	// Functional Utility: Call the core comparison logic.
	ret = cmp_files(argv[1],argv[2],precision);
	
	// Block Logic: Print the comparison result to the console.
	printf("%s %s %s %s\n", argv[0], argv[1], argv[2], (ret == 0 ? "OK" : "Incorrect results!"));
	return ret;
}

