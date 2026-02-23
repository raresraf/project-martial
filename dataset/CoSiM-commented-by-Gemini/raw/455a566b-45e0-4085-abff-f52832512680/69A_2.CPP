/**
 * @file 69A_2.CPP
 * @brief Solution to Codeforces problem 69A - Young Physicist.
 *
 * @algorithm
 * The problem asks to determine if a body is in equilibrium given a set of
 * `n` force vectors acting upon it.
 *
 * The core physical principle is that a body is in equilibrium if and only if
 * the net force (the vector sum of all individual forces) is zero.
 * For a 3D vector `F = (x, y, z)`, the vector sum is zero if and only if the
 * sum of the components along each axis is zero.
 *
 * Let F_i = (x_i, y_i, z_i) be the i-th force vector.
 * The body is in equilibrium if:
 *   Σx_i = 0
 *   Σy_i = 0
 *   Σz_i = 0
 *
 * This program implements this check directly:
 * 1. It initializes three variables to sum the x, y, and z components.
 * 2. It iterates `n` times, reading each force vector's components.
 * 3. It adds each component to the corresponding sum.
 * 4. After the loop, it checks if all three sums are zero.
 *
 * @complexity
 * - Time Complexity: O(N), where N is the number of force vectors, because
 *   the code iterates through the vectors once.
 * - Space Complexity: O(1), as it only uses a few variables to store the sums.
 */
#include <stdio.h>

int main() {
  /* n: the number of force vectors.
   * sum1, sum2, sum3: store the sum of forces along the x, y, and z axes, respectively.
   * a, b, c: store the x, y, z components of a single force vector.
   */
  int n, sum1 = 0, sum2 = 0, sum3 = 0, a, b, c;
  scanf("%d", &n);
  /* Block Logic: Loop `n` times to read all force vectors and update the component sums.
   * Pre-condition: `n` is the number of vectors, sums are 0.
   * Invariant: After `k` iterations, sum1, sum2, sum3 hold the component sums of the first `k` vectors.
   */
  while (n--) {
    scanf("%d %d %d", &a, &b, &c);
    sum1 += a; /* Accumulate the x-components. */
    sum2 += b; /* Accumulate the y-components. */
    sum3 += c; /* Accumulate the z-components. */
  }
  /* Block Logic: Check for equilibrium.
   * The body is in equilibrium if the net force vector is (0, 0, 0),
   * which means the sum of components along each axis must be 0.
   */
  if (sum1 == 0 && sum2 == 0 && sum3 == 0) {
    printf("YES\n");
  } else
    printf("NO\n");
}
