/**
 * @file 69A.CPP
 * @brief Solution to Codeforces problem 69A - Young Physicist.
 *        This program determines if a body is in equilibrium given a series of 3D force vectors
 *        acting on it. A body is in equilibrium if the sum of all force vectors acting on it
 *        is equal to the zero vector (i.e., all components sum to zero).
 *
 * Algorithm: Summation of Vector Components
 *   1. Reads `n`, the number of force vectors.
 *   2. Initializes three accumulators (`d`, `e`, `f`) to zero, representing the sum of
 *      x, y, and z components of all forces.
 *   3. Iterates `n` times, once for each force vector:
 *      a. Reads the x, y, and z components (`a`, `b`, `c`) of the current force.
 *      b. Adds `a` to `d`, `b` to `e`, and `c` to `f`.
 *   4. After processing all vectors, checks if all three sums (`d`, `e`, `f`) are zero.
 *   5. If all sums are zero, prints "YES" (equilibrium).
 *   6. Otherwise, prints "NO" (not in equilibrium).
 *
 * Time Complexity: O(N), where N is the number of force vectors, due to the single loop.
 * Space Complexity: O(1) for storing a few integer variables.
 */

#include <iostream>
using namespace std;

// Typedefs for various integer and floating-point types, common in competitive programming.
typedef long long int lli;
typedef long long ll;
typedef unsigned long long int ulli;
typedef double dbl;

int main(int argc, char const *argv[]) {
  // Optimizes C++ standard streams for faster input/output.
  ios_base::sync_with_stdio(false);

  ulli n; // Number of force vectors.
  cin >> n;

  lli a, b, c; // Components of a single force vector.
  lli d = 0, e = 0, f = 0; // Accumulators for the sum of x, y, z components.

  /**
   * Block Logic: Iterates through each of the `n` force vectors to accumulate their components.
   * Invariant: `d`, `e`, `f` hold the cumulative sums of x, y, and z components respectively.
   * Pre-condition: `n` is a non-negative integer.
   */
  while (n--) { // Loop executes `n` times.
    cin >> a >> b >> c; // Reads the x, y, and z components of the current force.
    d += a; // Accumulates x-components.
    e += b; // Accumulates y-components.
    f += c; // Accumulates z-components.
  }

  /**
   * Conditional Logic: Checks if the sum of all x, y, and z components is zero.
   * `!d` is equivalent to `d == 0` for integral types.
   */
  if (!d && !e && !f) { // If all three sums are zero.
    cout << "YES" << endl; // Prints "YES", indicating equilibrium.
  } else
    cout << "NO" << endl; // Prints "NO", indicating not in equilibrium.
  return 0;
}