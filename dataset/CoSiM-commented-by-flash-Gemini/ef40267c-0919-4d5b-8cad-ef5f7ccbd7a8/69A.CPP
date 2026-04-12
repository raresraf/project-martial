/**
 * @file 69A.CPP
 * @brief Determines if a system of forces is in equilibrium.
 *   This program sums N 3D force vectors and checks if the resultant force vector is a zero vector.
 * * Algorithm: Vector Summation.
 *   Reads N force vectors (x, y, z components) and computes the sum of components
 *   along each axis independently. The system is in equilibrium if and only if
 *   the sum of x-components, y-components, and z-components are all zero.
 * Time Complexity: O(N) where N is the number of force vectors.
 * Space Complexity: O(1) as only a few variables are used to store sums.
 */

#include <cmath>
#include <cstdio>

// Global variables to store the count of forces and individual force components, and their respective sums.
int n, x, y, z, sum_x, sum_y, sum_z;

/**
 * @brief Main function to compute the equilibrium of forces.
 * Functional Utility: Handles input, performs vector summation, and determines system equilibrium.
 */
int main() {
  scanf("%d", &n); // Reads the number of force vectors.

  /**
   * Block Logic: Iterates through each force vector and accumulates its components.
   * Invariant: At the start of each iteration, 'i' represents the current force vector index,
   *   and 'sum_x', 'sum_y', 'sum_z' hold the sums of components from previous vectors.
   */
  for (int i = 0; i < n; i++) {
    scanf("%d%d%d", &x, &y, &z); // Reads the x, y, and z components of the current force.

    sum_x += x; // Accumulates the x-component.
    sum_y += y; // Accumulates the y-component.
    sum_z += z; // Accumulates the z-component.
  }

  // Determines if the sum of absolute values of resultant force components is zero.
  // This is equivalent to checking if all three sums (sum_x, sum_y, sum_z) are zero.
  printf((std::abs(sum_x) + std::abs(sum_y) + std::abs(sum_z)) == 0 ? "YES"
                                                                    : "NO");

  return 0;
}