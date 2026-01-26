
/**
 * @file compare.c
 * @brief Utility for comparing two binary files containing double-precision floating-point numbers,
 *        interpreted as square matrices, based on a specified precision.
 * Algorithm: Memory-maps both files, calculates matrix dimensions, then iterates through elements
 *            performing an element-wise comparison with a given floating-point tolerance.
 * Time Complexity: O(N^2) where N is the dimension of the square matrix (number of elements).
 * Space Complexity: O(1) beyond the memory-mapped files, as comparison is done directly from mapped regions.
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <math.h>

// Macro for checking if two double-precision floating-point numbers are approximately equal
// within a specified error margin (err). Returns 0 if approximately equal, -1 otherwise.
#define check_err(a,b,err) ((fabs((a) - (b)) <= (err)) ? 0 : -1)

/**
 * @brief Compares two files containing square matrices of double-precision floating-point numbers.
 *
 * This function memory-maps the contents of two files, interprets them as square matrices,
 * and performs an element-wise comparison. If the absolute difference between corresponding
 * elements is within the specified 'precision', they are considered equal.
 *
 * @param file_path1 Path to the first file (matrix).
 * @param file_path2 Path to the second file (matrix).
 * @param precision The maximum allowed absolute difference between corresponding elements.
 * @return 0 if the files contain identical matrices within the given precision,
 *         -1 if the files differ in size, elements, or an error occurs during file operations.
 */
int cmp_files(char const *file_path1, char const *file_path2, double precision) {
	// fileInfo1, fileInfo2: Structures to hold file metadata (e.g., size).
	struct stat fileInfo1, fileInfo2;
	// mat1, mat2: Pointers to the memory-mapped regions of the files.
	double *mat1, *mat2;
	// i, j: Loop counters for matrix elements.
	// fd1, fd2: File descriptors for the input files.
	// N: Dimension of the square matrix (N x N).
	// ret: Return value of element comparison, also used as overall function return.
	int i, j, fd1, fd2, N, ret = 0;

	// Block Logic: Open both input files in read-only mode.
	fd1 = open(file_path1, O_RDONLY, (mode_t)0600);
	fd2 = open(file_path2, O_RDONLY, (mode_t)0600);

	// Block Logic: Retrieve file status (metadata) for both opened files.
	fstat(fd1, &fileInfo1);
	fstat(fd2, &fileInfo2);

	// Block Logic: Check if file sizes differ, indicating matrices of different dimensions.
	if(fileInfo1.st_size != fileInfo2.st_size) {
		printf("Files length differ\n");
		close(fd1);
		close(fd2);
		return -1;
	}

	// Block Logic: Memory-map the first file into virtual address space for direct access.
	mat1 = (double*) mmap(0, fileInfo1.st_size, PROT_READ, MAP_SHARED, fd1, 0);
	if (mat1 == MAP_FAILED)
	{
		close(fd1);
		close(fd2);
		printf("Error mmapping the first file");
		return -1;
	}

	// Block Logic: Memory-map the second file into virtual address space for direct access.
	mat2 = (double*) mmap(0, fileInfo2.st_size, PROT_READ, MAP_SHARED, fd2, 0);
	if (mat2 == MAP_FAILED)
	{
		munmap(mat1, fileInfo1.st_size); // Clean up: unmap the first file if second fails.
		close(fd1);
		close(fd2);
		printf("Error mmapping the second file");
		return -1;
	}

	// Calculate the dimension N of the square matrix.
	// Assumes file size is N*N*sizeof(double).
	N = sqrt(fileInfo1.st_size / sizeof(double));

	// Block Logic: Iterate through matrix elements to perform element-wise comparison.
	// Invariant: At the start of each outer loop, all elements up to row (i-1) have been compared and found to be within precision.
	for (i = 0; i < N; i++ ) {
		// Invariant: At the start of each inner loop, all elements up to column (j-1) in the current row 'i' have been compared.
		for (j = 0; j< N; j++) {
			// Compare current elements from both matrices with the specified precision.
			ret = check_err(mat1[i * N + j], mat2[i * N + j], precision); 
			if (ret != 0) {
				printf("Matrixes differ on index [%d, %d]. Expected %.8lf got %.8lf\n",
						i, j, mat1[i * N + j], mat2[i * N + j]);
				goto done; // Exit loops immediately on first difference.
			}
		}
	}

done:
	// Block Logic: Clean up resources by unmapping memory and closing file descriptors.
	munmap(mat1, fileInfo1.st_size);
	munmap(mat2, fileInfo2.st_size);

	close(fd1);
	close(fd2);

	return ret; // Return 0 for success (matrices are equal), -1 for failure.
}

/**
 * @brief Main function for the comparison utility.
 *
 * Parses command-line arguments for file paths and precision, then calls
 * `cmp_files` to perform the comparison and prints the result.
 *
 * @param argc The number of command-line arguments.
 * @param argv Array of command-line argument strings. Expected: [program_name, file1, file2, tolerance].
 * @return 0 on successful comparison with identical matrices, -1 otherwise.
 */
int main(int argc, const char **argv)
{    
	// precision: The floating-point tolerance for comparison, parsed from argv.
	double precision;
	// ret: Stores the return value of the cmp_files function.
	int ret = 0;

	// Block Logic: Check for correct number of command-line arguments.
	if(argc < 4) {
		printf("Usage: %s mat1 mat2 tolerance\n",argv[0]);
		exit(-1); // Exit if arguments are insufficient.
	}

	// Block Logic: Parse the precision value from the third command-line argument.
	sscanf(argv[3], "%lf", &precision);

	// Perform file comparison using the parsed arguments.
	ret = cmp_files(argv[1],argv[2],precision);
	
	// Print the overall result of the comparison.
	printf("%s %s %s %s\n", argv[0], argv[1], argv[2], (ret == 0 ? "OK" : "Incorrect results!"));
	return ret;
}
