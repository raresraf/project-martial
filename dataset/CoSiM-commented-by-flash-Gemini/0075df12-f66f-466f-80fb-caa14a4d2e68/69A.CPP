/**
 * @file 69A.CPP
 * @brief Solution for the "Young Physicist" problem from Codeforces.
 *
 * The problem asks to determine if a body is in equilibrium given a set of force
 * vectors acting on it. A body is in equilibrium if the sum of all force
 * vectors acting on it is equal to the zero vector. This means the sum of
 * all x-components, y-components, and z-components must each be zero.
 *
 * Problem Link: http://codeforces.com/problemset/problem/69/A
 *
 * Algorithm: Summation of vector components.
 * Time Complexity: O(N), where N is the number of force vectors.
 * Space Complexity: O(1), as only a constant number of variables are used
 *                  to store the running sums of the components.
 */

#include <iostream>
using namespace std;

int main() {
  int n; // Declares an integer to store the number of force vectors.
  // Reads the number of force vectors 'n' from standard input.
  cin >> n;

  int x, y, z;               // Variables to store the components of each force vector.
  int countX(0), countY(0), countZ(0); // Accumulators for the sums of x, y, and z components.

  /**
   * Block Logic: Reads 'n' force vectors and sums their respective components.
   * Invariant: At the end of each iteration, `countX`, `countY`, and `countZ`
   *            hold the cumulative sum of the components of all processed vectors.
   */
  for (int i = 0; i < n; i++) {
    // Reads the x, y, and z components of the current force vector.
    cin >> x >> y >> z;
    // Adds the current vector's components to the running sums.
    countX += x;
    countY += y;
    countZ += z;
  }

  /**
   * Block Logic: Checks if the sum of all force components is zero.
   * Functional Utility: Determines if the body is in a state of equilibrium.
   */
  if (countX == 0 && countY == 0 && countZ == 0) {
    // If all sums are zero, the body is in equilibrium.
    cout << "YES" << endl;
  } else {
    // Otherwise, the body is not in equilibrium.
    cout << "NO" << endl;
  }

  return 0;
}
