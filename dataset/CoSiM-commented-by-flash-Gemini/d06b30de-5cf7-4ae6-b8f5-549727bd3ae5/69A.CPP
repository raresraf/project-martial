/**
 * @file 69A.CPP
 * @brief This program determines if a system of 'n' forces acting on a body is in equilibrium.
 * * Algorithm: Vector sum calculation. It reads 'n' 3D force vectors and sums their respective X, Y, and Z components.
 *   The system is in equilibrium if and only if the sum of each component (X, Y, Z) is zero.
 * Time Complexity: O(N) where N is the number of force vectors, due to iterative summation.
 * Space Complexity: O(N) to store the 'n' force vectors.
 */

#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function to execute the force equilibrium check.
 * Functional Utility: Reads force vector components, calculates their sums, and determines if the system is in equilibrium.
 */
int main() {
  int n, i;
  // Variables a, b, c will store the sum of x, y, and z components, respectively.
  int a = 0, b = 0, c = 0;
  // Block Logic: Reads the number of force vectors 'n'.
  cin >> n;
  // Declares arrays to store the x, y, and z components of each force vector.
  int x[100], y[100], z[100];
  /**
   * Block Logic: Reads the x, y, and z components for each of the 'n' force vectors.
   * Pre-condition: 'n' is a positive integer representing the number of force vectors.
   * Invariant: After each iteration, `x[i]`, `y[i]`, `z[i]` contain the components of the (i+1)-th force vector.
   */
  for (i = 0; i < n; i++) {
    cin >> x[i] >> y[i] >> z[i];
  }
  /**
   * Block Logic: Sums the x, y, and z components of all force vectors.
   * Pre-condition: All force vectors' components have been read into arrays `x`, `y`, `z`.
   * Invariant: After each iteration, `a`, `b`, `c` accumulate the sums of x, y, z components up to the current vector.
   */
  for (i = 0; i < n; i++) {
    a += x[i];
    b += y[i];
    c += z[i];
  }
  /**
   * Block Logic: Checks if the system of forces is in equilibrium.
   * A system is in equilibrium if the sum of all x, y, and z components is zero.
   */
  if (a == 0 && b == 0 && c == 0)
    cout << "YES";
  else
    cout << "NO";

  return 0;
}