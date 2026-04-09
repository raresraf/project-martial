/**
 * @file 69A.CPP
 * @brief This program determines if a system of 'n' 3D force vectors is in equilibrium.
 * Equilibrium is achieved if the sum of all x-components, y-components, and z-components
 * of the forces individually equals zero. This is a fundamental concept in physics and
 * a common problem type in competitive programming.
 *
 * Algorithm: Vector sum and zero check.
 * 1. Read the number of vectors `n`.
 * 2. Initialize three accumulators (`x1`, `y1`, `z1`) to zero for each component.
 * 3. For each of the `n` vectors:
 *    a. Read its x, y, and z components (`x`, `y`, `z`).
 *    b. Add `x` to `x1`, `y` to `y1`, and `z` to `z1`.
 * 4. After processing all vectors, check if `x1`, `y1`, and `z1` are all zero.
 * 5. Print "YES" if all sums are zero, "NO" otherwise.
 * Time Complexity: O(N) where N is the number of vectors, as each vector is processed in constant time.
 * Space Complexity: O(1) for storing accumulators and input variables.
 */

#include <iostream>
using namespace std;
// Functional Utility: 'n' for the number of vectors.
// Functional Utility: 'x', 'y', 'z' for current vector components.
// Functional Utility: 'x1', 'y1', 'z1' are accumulators for the total sum of each component, initialized to 0.
int n, x, y, z, x1 = 0, y1 = 0, z1 = 0;

/**
 * @brief Main function of the program.
 * Reads a series of 3D vectors and checks if their sum is the zero vector.
 * @pre Input starts with an integer `n`, followed by `n` lines, each containing three integers (x, y, z components).
 * @post Prints "YES" if the net force is zero, "NO" otherwise.
 * @return 0 upon successful execution.
 */
int main() {
  // Functional Utility: Reads the number of force vectors.
  cin >> n;
  // Block Logic: Iterates through each of the 'n' force vectors.
  // Invariant: `x1`, `y1`, `z1` store the cumulative sum of components from vectors processed so far.
  for (int i = 0; i < n; i++) {
    // Functional Utility: Reads the x, y, and z components of the current vector.
    cin >> x >> y >> z;
    x1 += x; // Functional Utility: Accumulates the x-component.
    y1 += y; // Functional Utility: Accumulates the y-component.
    z1 += z; // Functional Utility: Accumulates the z-component.
  }
  // Block Logic: Checks if the total sum of all components is zero, indicating equilibrium.
  // Pre-condition: `x1`, `y1`, `z1` hold the final sums of all x, y, and z components respectively.
  if (x1 == 0 && y1 == 0 && z1 == 0) {
    cout << "YES"; // Functional Utility: Prints "YES" if the system is in equilibrium.
  } else {
    cout << "NO"; // Functional Utility: Prints "NO" if the system is not in equilibrium.
  }
  return 0;
}
