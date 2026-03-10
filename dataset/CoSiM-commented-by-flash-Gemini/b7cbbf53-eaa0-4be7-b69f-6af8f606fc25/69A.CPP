/**
 * @file 69A.CPP
 * @brief This program solves the "Young Physicist" problem (Codeforces 69A).
 *   It determines if a body is in equilibrium given a set of `t` force vectors.
 *   A body is in equilibrium if the sum of all forces acting on it is zero,
 *   meaning the sum of their x-components, y-components, and z-components must all be zero.
 *   Problem description: https://codeforces.com/problemset/problem/69/A
 * * Algorithm: Vector sum.
 *   It reads `t` force vectors (each with x, y, z components) and sums up all corresponding
 *   components separately. The body is in equilibrium if and only if all three sums (sum_x, sum_y, sum_z) are zero.
 * Time Complexity: O(T), where T is the number of force vectors, as it iterates through them once.
 * Space Complexity: O(1) for a few integer variables.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries.

using namespace std;

/**
 * @brief Main function to determine if a body is in equilibrium.
 * Functional Utility: Reads the number of force vectors and their components,
 *                     calculates the total force in each dimension, and prints
 *                     "YES" if the body is in equilibrium, "NO" otherwise.
 */
int main() {
  int t; // Stores the number of force vectors.
  // `a`, `b`, `c`: Temporary variables to read the x, y, and z components of each force vector.
  // `ca`, `cb`, `cc`: Accumulators for the sum of x, y, and z components across all force vectors.
  int a, b, c, ca = 0, cb = 0, cc = 0;
  // Block Logic: Reads the total number of force vectors, `t`.
  cin >> t;
  /**
   * Block Logic: Iterates `t` times to read each force vector and sum its components.
   * Pre-condition: `t` is a non-negative integer.
   * Invariant: After each iteration, `ca`, `cb`, `cc` accumulate the sums of x, y, z components
   *            up to the current vector.
   * Inline: `t--` decrements the counter and evaluates to true as long as `t` is greater than 0.
   */
  while (t--) {
    // Block Logic: Reads the x, y, and z components of the current force vector.
    cin >> a >> b >> c;
    ca += a; // Adds the x-component to the total x-sum.
    cb += b; // Adds the y-component to the total y-sum.
    cc += c; // Adds the z-component to the total z-sum.
  }
  /**
   * Conditional Logic: Checks if the body is in equilibrium.
   * A body is in equilibrium if the sum of all x, y, and z components is zero.
   * If `ca`, `cb`, and `cc` are all zero, it prints "YES". Otherwise, it prints "NO".
   */
  if (ca == 0 && cb == 0 && cc == 0)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}
