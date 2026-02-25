/**
 * @file raw/455a566b-45e0-4085-abff-f52832512680/69A_2.CPP
 * @brief Solution to Codeforces problem 69A - Young Physicist.
 *
 * @algorithm
 * The problem is based on a fundamental principle of physics: for a body to be
 * in equilibrium, the net force acting upon it must be zero. In a 3D coordinate
 * system, this means the vector sum of all forces is the zero vector.
 *
 * This condition is met if and only if the sum of the components of all force
 * vectors along each axis (x, y, and z) is individually zero.
 *
 * The algorithm implements this check directly:
 * 1. Read the total number of force vectors, `n`.
 * 2. Initialize three variables, `sum1`, `sum2`, and `sum3`, to 0. These will
 *    accumulate the sums of the x, y, and z components of the forces, respectively.
 * 3. Loop `n` times, once for each force vector.
 * 4. In each iteration, read the three components (x, y, z) of the current vector.
 * 5. Add each component to its corresponding total sum.
 * 6. After iterating through all the vectors, check if `sum1`, `sum2`, AND
 *    `sum3` are all equal to 0.
 * 7. If they are, the body is in equilibrium, and the program prints "YES".
 * 8. Otherwise, it prints "NO".
 *
 * @complexity
 * - Time Complexity: O(n), where n is the number of force vectors, because the
 *   program iterates through the vectors once.
 * - Space Complexity: O(1), as it only uses a few variables to store the sums,
 *   regardless of the number of vectors.
 */
#include <stdio.h>

int main() {
  int n; // Number of force vectors.
  // sum1, sum2, sum3: To store the sum of x, y, and z components respectively.
  int sum1 = 0, sum2 = 0, sum3 = 0;
  // a, b, c: To store the x, y, and z components of a single force vector.
  int a, b, c;

  // Read the number of force vectors.
  scanf("%d", &n);

  /**
   * @brief Loop to read each force vector and accumulate the component sums.
   */
  while (n--) {
    scanf("%d %d %d", &a, &b, &c);
    sum1 += a; // Accumulate x-components.
    sum2 += b; // Accumulate y-components.
    sum3 += c; // Accumulate z-components.
  }

  /**
   * @brief Check for equilibrium condition.
   * The net force is zero if and only if the sum of components on each
   * axis is zero.
   */
  if (sum1 == 0 && sum2 == 0 && sum3 == 0) {
    printf("YES\n");
  } else {
    printf("NO\n");
  }

  // The original code is missing a return statement, which is acceptable in C
  // for main, where it defaults to 0 if execution completes successfully.
}
