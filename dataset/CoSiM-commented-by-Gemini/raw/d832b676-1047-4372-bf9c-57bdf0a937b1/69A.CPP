/**
 * @file 69A.CPP
 * @brief Solution to a competitive programming problem checking for equilibrium of forces.
 * This program reads `n` 3D force vectors, stores their components in arrays,
 * and then determines if the sum of all x, y, and z components individually
 * equals zero, indicating a state of equilibrium.
 *
 * Algorithm: Array-based summation of vector components.
 * Time Complexity: O(N) where N is the number of force vectors.
 * Space Complexity: O(N) for storing vector components in arrays.
 */
#include <iostream>
using namespace std;
/**
 * @brief Main function of the program.
 * Reads the number of force vectors `n`, then processes `n` vectors by storing
 * their x, y, and z components into arrays. It then calculates the sum of
 * all x, y, and z components separately to check for equilibrium.
 * @return `0` upon successful execution.
 */
int main() {
  int n;
  cin >> n;
  // Functional Utility: Declares variable-length arrays (VLAs) to store x, y, and z components of `n` force vectors.
  // This allows for dynamic sizing of arrays based on input `n`.
  int x[n], y[n], z[n];
  /**
   * @brief Loop to read `n` force vectors into arrays.
   * Invariant: `i` ranges from 0 to `n-1`, populating the `x`, `y`, and `z` arrays
   * with component values for each force vector.
   */
  for (int i = 0; i < n; i++) {
    cin >> x[i] >> y[i] >> z[i];
  }
  int a, b, c;
  a = 0;
  b = 0;
  c = 0;
  /**
   * @brief Loop to calculate the sum of each component across all force vectors.
   * Invariant: `i` ranges from 0 to `n-1`, accumulating the sum of x-components into `a`,
   * y-components into `b`, and z-components into `c`.
   */
  for (int i = 0; i < n; i++) {
    a = a + x[i];
    b = b + y[i];
    c = c + z[i];
  }
  /**
   * @brief Checks for the equilibrium condition.
   * An object is in equilibrium if the sum of all x, y, and z components are all zero.
   * The output is "YES" if in equilibrium, otherwise "NO".
   */
  if (a == 0) {
    if (b == 0 && c == 0) {
      cout << "YES" << endl;
    }
  } else {
    cout << "NO" << endl;
  }
  return 0;
}
