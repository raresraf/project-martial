/**
 * @file 69A.CPP
 * @brief Solution to the "Young Physicist" problem (Codeforces 69A), checking for force equilibrium.
 *
 * @details This program determines if a set of 3D force vectors results in a net force of zero.
 * A body is in equilibrium if the vector sum of all forces acting on it is zero.
 * This translates to the sum of the components along each axis (x, y, z) being zero.
 *
 * @algorithm The program first reads the number of vectors, `n`. It then reads the `n` vectors,
 * storing their x, y, and z components in separate arrays. It calculates the sum of all
 * x-components, y-components, and z-components. Finally, it checks if all three sums are
 * equal to zero. If they are, the system is in equilibrium ("YES"); otherwise, it is not ("NO").
 *
 * @complexity
 * Time Complexity: O(N), where N is the number of vectors, due to the linear scan for input and summation.
 * Space Complexity: O(N), to store the x, y, and z components of the N vectors.
 */
#include <stdio.h>
int main() {
  int i, n, x[105], y[105], z[105], sumx = 0, sumy = 0, sumz = 0;
  scanf("%d", &n);

  // Read the n vectors from standard input.
  for (i = 0; i < n; i++)
    scanf("%d %d %d", &x[i], &y[i], &z[i]);

  /**
   * @brief Block Logic: Calculates the sum of components for each axis.
   * @invariant At the end of iteration `i`, `sumx`, `sumy`, and `sumz` contain the
   * component-wise sum of the first `i+1` vectors.
   */
  for (i = 0; i < n; i++) {
    sumx += x[i];
    sumy += y[i];
    sumz += z[i];
  }

  /**
   * @brief Block Logic: Checks for equilibrium condition.
   * A system is in equilibrium if the net force vector is (0, 0, 0).
   */
  if (sumx == 0 && sumy == 0 && sumz == 0)
    printf("YES");
  else
    printf("NO");
  return 0;
}