/**
 * @file 69A.CPP
 * @brief Solution for Codeforces Problem 69A - "Young Physicist".
 * @details This program checks if a body is in equilibrium by summing up all the force
 * vectors acting on it. For the body to be in equilibrium, the resultant force
 * vector must be the zero vector (0, 0, 0).
 *
 * @author (unknown)
 *
 * @algorithm
 * 1. Read the number of applied forces, `n`.
 * 2. Initialize three variables, `x`, `y`, and `z`, to 0. These will store the
 *    sum of the components of all force vectors along the respective axes.
 * 3. Loop `n` times to read each force vector.
 * 4. In each iteration, read the three components of a force and add them to the
 *    running sums `x`, `y`, and `z`.
 * 5. After the loop, `x`, `y`, and `z` represent the components of the resultant force vector.
 * 6. Check if `x`, `y`, and `z` are all simultaneously zero.
 * 7. If they are, the body is in equilibrium, so print "YES". Otherwise, print "NO".
 *
 * @complexity
 * Time Complexity: O(n), where `n` is the number of forces, due to the single summation loop.
 * Space Complexity: O(1), as it uses a fixed number of variables.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  /**
   * Functional Utility: Decouples C++ streams from C streams and unties cin/cout
   * for faster I/O, a common optimization in competitive programming.
   */
  ios::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(NULL);

  int n, x = 0, y = 0, z = 0, a, b, c;
  cin >> n;

  /**
   * Block Logic: This loop iterates through all 'n' forces, accumulating the
   * components of each force vector into the sum variables x, y, and z.
   */
  while (n--) {
    cin >> a >> b >> c;
    x += a;
    y += b;
    z += c;
  }

  /**
   * Block Logic: Checks for equilibrium. If the sum of forces along each axis is zero,
   * the net force is zero, and the body is in equilibrium. A ternary operator
   * is used for a concise conditional output.
   */
  (x == 0 && y == 0 && z == 0) ? cout << "YES\n" : cout << "NO\n";
}