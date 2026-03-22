/**
 * @file 69A.CPP
 * @brief Solution to Codeforces problem 69A - Young Physicist.
 *        This program determines if a body is in equilibrium given a series of forces acting on it.
 *        A body is in equilibrium if the sum of all force vectors acting on it is equal to zero.
 *
 * Algorithm:
 *   1. Reads an integer `n`, representing the number of force vectors.
 *   2. Initializes three accumulators (`x1`, `y1`, `z1`) to zero for the sum of x, y, and z components, respectively.
 *   3. For `n` times, it reads the x, y, and z components of a force vector.
 *   4. Each component is added to its respective accumulator (`x1 += x`, `y1 += y`, `z1 += z`).
 *   5. After processing all `n` vectors, it checks if all three accumulators (`x1`, `y1`, `z1`) are zero.
 *   6. If all sums are zero, it indicates the body is in equilibrium, and "YES" is printed.
 *   7. Otherwise, the body is not in equilibrium, and "NO" is printed.
 *
 * Time Complexity: O(n), where `n` is the number of force vectors, due to the loop.
 * Space Complexity: O(1) for storing a few integer variables.
 */

#include <iostream>
using namespace std;
int n, x, y, z, x1 = 0, y1 = 0, z1 = 0;
int main() {
  /**
   * @brief Main function that solves the "Young Physicist" problem.
   *
   * Reads a series of 3D force vectors and determines if their resultant sum is zero,
   * indicating a state of equilibrium.
   *
   * @return int Returns 0 upon successful execution.
   */
  cin >> n;
  /**
   * Block Logic: Accumulates the sum of x, y, and z components of `n` force vectors.
   * Invariant: `x1`, `y1`, `z1` store the cumulative sums of components up to the current iteration.
   * Pre-condition: `n` is a positive integer representing the number of force vectors.
   */
  for (int i = 0; i < n; i++) {
    cin >> x >> y >> z;
    x1 += x;
    y1 += y;
    z1 += z;
  }
  /**
   * Block Logic: Evaluates if the system is in equilibrium.
   * Pre-condition: `x1`, `y1`, `z1` contain the sums of all x, y, and z components.
   * Post-condition: Prints "YES" if all sums are zero, "NO" otherwise.
   */
  if (x1 == 0 && y1 == 0 && z1 == 0) {
    cout << "YES";
  } else {
    cout << "NO";
  }
}
