/**
 * @file 69A.CPP
 * @brief Determines if a body is in equilibrium given a set of force vectors.
 *
 * The program reads a set of 3D force vectors. It checks if the vector sum of
 * all these forces is zero. If it is, the body is in equilibrium.
 *
 * @b Algorithm: The program reads `n` force vectors. It maintains a running
 *    sum for each of the x, y, and z components of the vectors. After summing
 *    up all the vectors, it checks if the sums of all three components are zero.
 *
 * @b Time_Complexity: O(N), where N is the number of force vectors, because it
 *    iterates through the vectors once.
 * @b Space_Complexity: O(1), as it uses a fixed number of variables to store the sums.
 */
#include <cmath>
#include <cstdio>

// Note: Using global variables is generally not recommended.
int n, x, y, z, sum_x, sum_y, sum_z;
int main() {
  scanf("%d", &n);

  /**
   * @brief This loop reads each force vector and adds its components to the total sums.
   */
  for (int i = 0; i < n; i++) {
    scanf("%d%d%d", &x, &y, &z);

    sum_x += x;
    sum_y += y;
    sum_z += z;
  }

  /**
   * @brief Checks for equilibrium.
   * A body is in equilibrium if the net force is zero. In 3D, this means the
   * sum of forces along each axis (x, y, and z) must be zero. The absolute values
   * are used here as a shortcut; if sum_x, sum_y, and sum_z are all zero, their
   * absolute sum will be zero.
   * The result is printed using a ternary operator inside printf.
   */
  printf((std::abs(sum_x) + std::abs(sum_y) + std::abs(sum_z)) == 0 ? "YES"
                                                                    : "NO");

  return 0;
}