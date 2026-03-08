/**
 * @file 69A.CPP
 * @brief Solution for the "Young Physicist" problem on Codeforces.
 *
 * @details
 * This program determines if a body is in equilibrium under the influence of
 * several forces. In physics, a body is in equilibrium if the vector sum of all
 * forces acting on it is zero. The program reads a number of 3D force vectors,
 * calculates the sum of their x, y, and z components, and checks if all three
 * sums are zero.
 *
 * @section
 * Algorithm: Iterative summation of vector components.
 * Time Complexity: O(N), where N is the number of force vectors.
 * Space Complexity: O(1).
 */
#include <stdio.h>

int main() {
  // n: the number of forces. x, y, z: components of a single force vector.
  // Using 'short' is a minor optimization assuming component values are small.
  short x, y, z, n;
  // sumx, sumy, sumz: accumulators for the components of the net force vector.
  int sumx = 0, sumy = 0, sumz = 0;
  scanf("%hd", &n);
  /**
   * @brief This loop reads each force vector and adds it to the net force.
   *
   * Invariant: At the start of each iteration, sumx, sumy, and sumz hold the
   * vector sum of all forces processed so far.
   */
  while (n--) {
    scanf("%hd %hd %hd", &x, &y, &z);
    sumx += x;
    sumy += y;
    sumz += z;
  }
  /**
   * @brief Check for equilibrium.
   *
   * Logic: The body is in equilibrium if and only if the net force along each
   * axis is zero. This condition checks if the total x, y, and z sums are all
   * equal to 0.
   */
  if (sumx == 0 && sumy == 0 && sumz == 0)
    printf("%s", "YES");
  else
    printf("%s", "NO");

  return 0;
}
