/**
 * @file 69A.CPP
 * @brief Solves the "Young Physicist" problem from Codeforces (Problem 69A).
 * @details This program determines if a body is in equilibrium given a set of
 * force vectors acting upon it. A body is in equilibrium if the vector sum of
 * all forces is zero.
 *
 * @algorithm
 * 1. Read the number of force vectors, `n`.
 * 2. Initialize three variables, `x`, `y`, and `z`, to 0. These will store the
 *    sum of the components for each axis.
 * 3. Loop `n` times to read each force vector.
 * 4. In each iteration, read the x, y, and z components of a vector and add
 *    them to the corresponding sum variables.
 * 5. After the loop, check if the sums of all three components (`x`, `y`, `z`)
 *    are simultaneously zero.
 * 6. If they are all zero, the body is in equilibrium, so print "YES".
 * 7. Otherwise, print "NO".
 *
 * @time_complexity O(n), where `n` is the number of force vectors.
 * @space_complexity O(1).
 */
#include <iostream>

using namespace std;

int main() {
  int n; // The number of force vectors.
  int x = 0, y = 0, z = 0; // To store the sum of components along each axis.
  cin >> n;

  /**
   * @block_logic Read each force vector and accumulate the components.
   * @pre `n` is the number of vectors to read.
   * @post `x`, `y`, and `z` will hold the sum of all vector components.
   */
  for (int i = 0; i < n; ++i) {
    int a, b, c; // Components of a single force vector.
    cin >> a >> b >> c;
    x += a;
    y += b;
    z += c;
  }

  // Check if the net force vector is zero.
  if (x == 0 && y == 0 && z == 0)
    cout << "YES";
  else
    cout << "NO";
}