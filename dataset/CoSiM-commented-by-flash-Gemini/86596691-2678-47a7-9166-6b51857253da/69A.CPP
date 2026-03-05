/**
 * @file 69A.CPP
 * @brief Solution to a problem that determines if a body is in equilibrium
 *        under the action of N forces. Each force is given by its three
 *        components (Fx, Fy, Fz). The body is in equilibrium if the sum of
 *        all force components in each dimension (x, y, z) is zero.
 *
 * Algorithm: Summation of vector components.
 *            The algorithm reads N sets of three integer components (x, y, z)
 *            and maintains a running sum for each dimension. After reading
 *            all N sets, it checks if all three sums are simultaneously zero.
 * Time Complexity: O(N), where N is the number of force vectors.
 *                  The program iterates N times, performing constant-time
 *                  addition operations in each iteration.
 * Space Complexity: O(1), as only a few variables are used to store sums.
 */

#include <iostream>

int main() {
  // Functional Utility: Unties C++ standard streams from C standard streams
  // and disables synchronization, optimizing high-throughput I/O.
  std::ios::sync_with_stdio(false);
  // Functional Utility: Initializes variables to store the number of force vectors (n)
  // and temporary components (tmp_x, tmp_y, tmp_z), along with cumulative sums (x, y, z).
  int n, tmp_x, tmp_y, tmp_z, x = 0, y = 0, z = 0;
  // Block Logic: Reads the total number of force vectors (n) from standard input.
  std::cin >> n;
  /**
   * Block Logic: Iterates N times to read each force vector's components and
   *              accumulate their sums.
   * Invariant: In each iteration, `n` decreases, and `tmp_x, tmp_y, tmp_z` hold
   *            the components of the current force vector. `x, y, z` accumulate
   *            the sums of components read so far.
   */
  while (n--) {
    // Block Logic: Reads the x, y, and z components of the current force vector.
    std::cin >> tmp_x >> tmp_y >> tmp_z;
    // Block Logic: Adds the current force vector's components to the respective
    //              cumulative sums.
    x += tmp_x;
    y += tmp_y;
    z += tmp_z;
  }
  // Post-condition: Checks if the sum of forces in all three dimensions is zero.
  // Functional Utility: Determines if the body is in equilibrium based on the
  // final accumulated force components.
  if (x == 0 && y == 0 && z == 0)
    // Block Logic: If all sums are zero, prints "YES" indicating equilibrium.
    std::cout << "YES" << std::endl;
  else
    // Block Logic: Otherwise, prints "NO" indicating the body is not in equilibrium.
    std::cout << "NO" << std::endl;

  return 0;
}