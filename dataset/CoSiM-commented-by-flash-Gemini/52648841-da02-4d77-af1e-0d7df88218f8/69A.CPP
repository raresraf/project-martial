/**
 * @file 69A.CPP
 * @brief Solution to Codeforces Problem 69A: "Young Physicist".
 *        This program determines if a system of forces acting on an object is in equilibrium.
 *        Given a number of forces, each specified by its X, Y, and Z components,
 *        the program calculates the sum of components in each dimension. If all sums are zero,
 *        the system is in equilibrium.
 *
 * Algorithm: Vector summation. Reads N force vectors and sums their respective X, Y, and Z components.
 *            Checks if the resultant vector is the zero vector.
 * Time Complexity: O(N) where N is the number of force vectors, due to a single pass iteration
 *                  to sum the components.
 * Space Complexity: O(1) as only a few variables are used to store sums and temporary inputs.
 */

#include <iostream>
using namespace std;

int main() {
  /**
   * Functional Utility: Manages the overall process of reading the number of forces,
   *                     iteratively summing their components, and determining if the system is in equilibrium.
   */
  int n; // Declares an integer to store the number of force vectors.
  cin >> n; // Reads the number of force vectors from standard input.
  int a, b, c; // Declares integers to store the X, Y, and Z components of a single force vector.
  int x, y, z; // Declares integers to store the cumulative sums of the X, Y, and Z components of all force vectors.
  x = y = z = 0; // Initializes the cumulative sums to zero.
  /**
   * Block Logic: Iterates 'n' times to read each force vector and accumulate its components.
   * Pre-condition: `n` contains the number of force vectors to process; `x`, `y`, `z` are initialized to 0.
   * Invariant: After each iteration, `x`, `y`, `z` hold the correct sums of the X, Y, and Z components
   *            of all force vectors processed so far.
   */
  for (int i = 0; i < n; i++) {
    cin >> a >> b >> c; // Reads the X, Y, and Z components of the current force vector.
    x += a;             // Adds the current X component to the total X sum.
    y += b;             // Adds the current Y component to the total Y sum.
    z += c;             // Adds the current Z component to the total Z sum.
  }
  /**
   * Block Logic: Checks if the system of forces is in equilibrium.
   * Pre-condition: `x`, `y`, `z` contain the total sums of the X, Y, and Z components of all forces.
   * Invariant: Prints "YES" if all component sums are zero (indicating equilibrium); otherwise, prints "NO".
   */
  if ((x == 0) && (y == 0) && (z == 0)) // Check if all component sums are zero.
    cout << "YES\n";                     // Output "YES" if the system is in equilibrium.
  else
    cout << "NO\n"; // Output "NO" if the system is not in equilibrium.
  // Implicit return 0; due to successful program completion.
}
