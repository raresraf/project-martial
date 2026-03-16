/**
 * @file 69A.CPP
 * @brief Solution to Codeforces problem 69A - "Young Physicist".
 * @details The program determines if a body is in equilibrium by calculating the
 *          vector sum of all forces acting on it. If the net force vector is (0,0,0),
 *          the body is in equilibrium.
 *
 * Algorithm:
 * 1. Read the number of force vectors, `n`.
 * 2. Initialize total force components (`tx`, `ty`, `tz`) for each axis to zero.
 * 3. Loop `n` times:
 *    a. Read the x, y, and z components of a single force vector.
 *    b. Add these components to the running totals for each axis.
 * 4. After the loop, check if the total force on each axis is zero.
 * 5. If `tx`, `ty`, and `tz` are all zero, print "YES". Otherwise, print "NO".
 *
 * Time Complexity: O(N), where N is the number of force vectors.
 * Space Complexity: O(1).
 */
#include <iostream>
using namespace std;
int main() {
  int n;
  cin >> n;
  int x, y, z;
  // `tx`, `ty`, `tz` are the totals for the x, y, and z components of the forces.
  int tx = 0, ty = 0, tz = 0;

  /**
   * @brief Main loop to read `n` vectors and calculate the net force.
   * Invariant: After each iteration, `tx`, `ty`, and `tz` hold the vector
   * sum of all forces processed so far.
   */
  while (n--) {
    cin >> x >> y >> z;
    // Add the components of the current vector to the total sums.
    // The comma operator is used to sequence these operations on a single line.
    tx += x, ty += y, tz += z;
  }

  /**
   * @brief Check for the equilibrium condition.
   * If the net force along all axes is zero, the object is not moving.
   */
  if (tx == 0 && ty == 0 && tz == 0)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}
