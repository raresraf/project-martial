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

using namespace std;

int main() {
  // Functional Utility: Initializes `n` for the number of force vectors,
  // `x, y, z` for individual components, and `xsum, ysum, zsum` for cumulative sums.
  int n, x, y, z, xsum(0), ysum(0), zsum(0);
  // Block Logic: Reads the total number of force vectors `n` from standard input.
  cin >> n;
  /**
   * Block Logic: Iterates `n` times to read each force vector's components and
   *              accumulate their sums.
   * Invariant: `n` decreases with each iteration, and `xsum, ysum, zsum`
   *            accumulate the components of forces processed so far.
   */
  while (n--) {
    // Block Logic: Reads the x, y, and z components of the current force vector.
    cin >> x >> y >> z;
    // Functional Utility: Adds the current components to the respective cumulative sums.
    xsum += x;
    ysum += y;
    zsum += z;
  }
  // Post-condition: Checks if the sum of forces in all three dimensions is zero.
  // Functional Utility: Determines if the body is in equilibrium based on the
  // final accumulated force components.
  if (xsum == 0 && ysum == 0 && zsum == 0) {
    // Block Logic: If all sums are zero, prints "YES" followed by a newline.
    cout << "YES\n";
  } else {
    // Block Logic: Otherwise, prints "NO" followed by a newline.
    cout << "NO\n";
  }
  return 0;
}
