/**
 * @file 69A.CPP
 * @brief Checks if a system of 3D vectors is in equilibrium.
 *
 * This program reads a specified number of 3D vectors and determines if their
 * vector sum is the zero vector (i.e., the sum of the x, y, and z components
 * are all independently zero). If the sum is the zero vector, it prints "YES",
 * otherwise it prints "NO". This is a classic statics problem.
 *
 * Algorithm:
 * 1. Read an integer `n`, which is the total number of vectors.
 * 2. Read the `n` vectors, storing their x, y, and z components.
 * 3. Initialize three variables `sumx`, `sumy`, and `sumz` to zero.
 * 4. Iterate through the stored vectors, accumulating the sums of their
 *    respective components into `sumx`, `sumy`, and `sumz`.
 * 5. After the loop, check if `sumx`, `sumy`, and `sumz` are all zero.
 * 6. If they are, the system is in equilibrium, and the program prints "YES".
 * 7. Otherwise, it prints "NO".
 *
 * Time Complexity: O(N), where N is the number of vectors. The program iterates
 * through the vectors twice: once to read them and once to sum their components.
 * Space Complexity: O(N), for storing the x, y, and z components of the N vectors in arrays.
 */
#include <stdio.h>

int main() {
  int i, n, x[105], y[105], z[105], sumx = 0, sumy = 0, sumz = 0;
  // Read the number of vectors.
  scanf("%d", &n);

  /**
   * @brief Reads the components of `n` vectors from standard input.
   * Pre-condition: `n` is the number of vectors to be read.
   * Post-condition: Arrays x, y, and z are populated with vector components.
   */
  for (i = 0; i < n; i++)
    scanf("%d %d %d", &x[i], &y[i], &z[i]);

  /**
   * @brief Calculates the sum of all x, y, and z components.
   * Invariant: At the start of each iteration `i`, `sumx`, `sumy`, `sumz` hold the
   * sum of components for the first `i` vectors.
   */
  for (i = 0; i < n; i++) {
    sumx += x[i];
    sumy += y[i];
    sumz += z[i];
  }

  /**
   * @brief Checks if the total vector sum is zero.
   * This is the condition for static equilibrium.
   */
  if (sumx == 0 && sumy == 0 && sumz == 0)
    printf("YES");
  else
    printf("NO");

  return 0;
}