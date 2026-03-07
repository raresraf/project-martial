/**
 * @file 69A.CPP
 * @brief Solves the "Young Physicist" problem.
 *
 * @details
 * The program determines if a body is in equilibrium under the influence of
 * several force vectors. A body is in equilibrium if the vector sum of all
 * forces acting on it is zero. This means the sum of the x, y, and z components
 * of all force vectors must individually be zero.
 *
 * Algorithm:
 * 1. Read the number of force vectors, n.
 * 2. Initialize three variables (sum_x, sum_y, sum_z) to zero to store the net force components.
 * 3. Loop n times:
 *    a. Read the x, y, and z components of a single force vector.
 *    b. Add these components to the respective sum variables.
 * 4. After the loop, check if sum_x, sum_y, and sum_z are all equal to zero.
 * 5. If they are, the body is in equilibrium, print "YES". Otherwise, print "NO".
 *
 * Time Complexity: O(N), where N is the number of force vectors.
 * Space Complexity: O(1).
 */
#include <iostream>

int main() {
  // Disable synchronization with C-style I/O for faster input.
  std::ios::sync_with_stdio(false);
  
  int n; // Number of force vectors.
  // Temporary variables to read vector components.
  int tmp_x, tmp_y, tmp_z;
  // Variables to store the sum of components for each axis.
  int x = 0, y = 0, z = 0;
  
  std::cin >> n;
  
  /**
   * @brief Loop to read each force vector and accumulate the component sums.
   */
  while (n--) {
    std::cin >> tmp_x >> tmp_y >> tmp_z;
    x += tmp_x;
    y += tmp_y;
    z += tmp_z;
  }
  
  /**
   * @brief Check for equilibrium condition.
   * The vector sum is zero if and only if the sum of each component is zero.
   */
  if (x == 0 && y == 0 && z == 0)
    std::cout << "YES" << std::endl;
  else
    std::cout << "NO" << std::endl;

  return 0;
}