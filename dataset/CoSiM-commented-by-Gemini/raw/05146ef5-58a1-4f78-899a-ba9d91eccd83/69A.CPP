/**
 * @file 69A.CPP
 * @brief Solution for Codeforces Problem 69A, "Young Physicist".
 * @details This program determines if a body is in equilibrium given a set of
 * force vectors acting upon it. A body is in equilibrium if and only if the
 * vector sum of all forces is zero.
 *
 * @section Algorithm
 * The algorithm iterates through each force vector and calculates the sum of its
 * x, y, and z components separately. After processing all vectors, it checks if
 * the sum of components for each axis is equal to zero. If all three sums are
 * zero, the net force is zero, and the body is in equilibrium.
 *
 * @section Complexity
 * - Time Complexity: O(N), where N is the number of force vectors, as the program
 *   makes a single pass through the input to sum the components.
 * - Space Complexity: O(1), as the program only uses a fixed number of variables
 *   to store the component sums, independent of the input size.
 */
#include <iostream>

using namespace std;

/**
 * @brief Main function to solve the equilibrium problem.
 *
 * Reads the number of force vectors (n), then iterates n times to read each
 * vector's x, y, and z components. It accumulates the sums of these components
 * and finally checks if the total sum for each axis is zero to determine if
 * the body is in equilibrium.
 */
int main() {
  int n, x, y, z;
  // Functional Utility: sux, suy, suz store the net force components along the x, y, and z axes.
  int sux = 0, suy = 0, suz = 0;
  cin >> n;
  /**
   * @brief Iterates through each of the n force vectors to calculate the net force.
   * @invariant At the beginning of each iteration, sux, suy, and suz hold the
   * sum of force components from all previously processed vectors.
   */
  for (int i = 0; i < n; ++i) {
    cin >> x >> y >> z;
    sux += x;
    suy += y;
    suz += z;
  }
  /**
   * @brief Checks the equilibrium condition.
   * For a body to be in equilibrium, the net force along each of the three
   * perpendicular axes (x, y, z) must be zero.
   */
  if (sux == 0 && suy == 0 && suz == 0)
    cout << "YES"
         << "\n";
  else
    cout << "NO"
         << "\n";
  return 0;
}