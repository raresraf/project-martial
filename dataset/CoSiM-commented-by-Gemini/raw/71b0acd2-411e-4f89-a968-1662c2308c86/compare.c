/**
 * @file compare.c
 * @brief A utility program to compare two matrices stored in binary files.
 *
 * This program takes two file paths, each containing a square matrix of doubles in binary format,
 * and a tolerance value. It compares the two matrices element by element to check if they
 * are identical within the given tolerance. The comparison is performed efficiently by
 * memory-mapping the files.
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
 * @def check_err(a, b, err)
 * @brief A macro to check if two double-precision numbers are close enough within a given tolerance.
 *
 * @param a The first number.
 * @param b The second number.
 * @param err The tolerance (maximum allowed difference).
 * @return 0 if the absolute difference is within the tolerance, -1 otherwise.
 */
#define check_err(a,b,err) ((fabs((a) - (b)) <= (err)) ? 0 : -1)


/**
 * @brief Compares two matrices from binary files using memory mapping.
 *
 * @param file_path1 Path to the first binary file containing a matrix.
 * @param file_path2 Path to the second binary file containing a matrix.
 * @param precision The tolerance for floating-point comparison.
 * @return 0 if the matrices are identical within the given precision, -1 otherwise.
 *
 * @note This function uses mmap for efficient file reading, which maps the file content
 *       directly into the process's address space. This avoids the overhead of repeated
 *       read() system calls.
 */
int cmp_files(char const *file_path1, char const *file_path2, double precision) {
	struct stat fileInfo1, fileInfo2;
	double *mat1, *mat2;
	int i, j, fd1, fd2, N, ret = 0;

	// Open both files for reading.
	fd1 = open(file_path1, O_RDONLY, (mode_t)0600);
	fd2 = open(file_path2, O_RDONLY, (mode_t)0600);

	// Get file stats to check their sizes.
	fstat(fd1, &fileInfo1);
	fstat(fd2, &fileInfo2);

	// Pre-condition: If file sizes are different, the matrices cannot be identical.
	if(fileInfo1.st_size != fileInfo2.st_size) {
		printf("Files length differ
");
		close(fd1);
		close(fd2);
		return -1;
	}

	// Memory-map the first file. This is an efficient way to access file content.
	mat1 = (double*) mmap(0, fileInfo1.st_size, PROT_READ, MAP_SHARED, fd1, 0);
	if (mat1 == MAP_FAILED)
	{
		close(fd1);
		close(fd2);
		printf("Error mmapping the first file");
		return -1;
	}

	// Memory-map the second file.
	mat2 = (double*) mmap(0, fileInfo2.st_size, PROT_READ, MAP_SHARED, fd2, 0);
	if (mat2 == MAP_FAILED)
	{
		munmap(mat1, fileInfo1.st_size);
		close(fd1);
		close(fd2);
		printf("Error mmapping the second file");
		return -1;
	}

	// Assumption: The matrices are square. Calculate the dimension N.
	N = sqrt(fileInfo1.st_size / sizeof(double));

	// Block Logic: Iterate through the matrices and compare each element.
	// Invariant: All elements checked so far are within the tolerance.
	for (i = 0; i < N; i++ ) {
		for (j = 0; j< N; j++) {
			ret = check_err(mat1[i * N + j], mat2[i * N + j], precision); 
			if (ret != 0) {
				printf("Matrixes differ on index [%d, %d]. Expected %.8lf got %.8lf
",
						i, j, mat1[i * N + j], mat2[i * N + j]);
				goto done;
			}
		}
	}

done:
	// Cleanup: Unmap the memory regions and close the file descriptors.
	munmap(mat1, fileInfo1.st_size);
	munmap(mat2, fileInfo2.st_size);

	close(fd1);
	close(fd2);

	return ret;
}

/**
 * @brief The main function of the program.
 *
 * @param argc The number of command-line arguments.
 * @param argv An array of command-line arguments.
 * @return 0 on success (matrices are identical), -1 on failure.
 *
 * @note The program expects three command-line arguments: the paths to the two
 *       matrix files and the comparison tolerance.
 */
int main(int argc, const char **argv)
{    
	double precision;
	int ret = 0;

	if(argc < 4) {
		printf("Usage: %s mat1 mat2 tolerance
",argv[0]);
		exit(-1);
	}

	sscanf(argv[3], "%lf", &precision);

	ret = cmp_files(argv[1],argv[2],precision);
	
	printf("%s %s %s %s
", argv[0], argv[1], argv[2], (ret == 0 ? "OK" : "Incorrect results!"));
	return ret;
}
