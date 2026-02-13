/**
 * @file 69A.CPP
 * @brief Solution for the "Young Physicist" problem (Codeforces 69A).
 * @author competitive-programmer
 *
 * @details This program determines if a system is in equilibrium by checking if
 * the vector sum of a given set of 3D force vectors is the zero vector.
 *
 * Algorithm (Optimal Single-Pass):
 * 1. Initialize three variables, `xsum`, `ysum`, and `zsum`, to 0. These will
 *    store the net force along each axis.
 * 2. Read the number of force vectors, `n`.
 * 3. Loop `n` times. In each iteration:
 *    a. Read the x, y, and z components of a single force vector.
 *    b. Immediately add these components to their respective sum variables.
 * 4. After the loop, if `xsum`, `ysum`, and `zsum` are all zero, the system is
 *    in equilibrium. Print "YES". Otherwise, print "NO".
 *
 * This approach is optimal as it processes the input in a single pass and uses
 * constant extra space.
 *
 * Time Complexity: O(n), where n is the number of vectors.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // n: number of vectors.
  // x, y, z: components of a single vector.
  // xsum, ysum, zsum: running sums of the components, initialized to 0.
  int x, y, z, n, xsum(0), ysum(0), zsum(0);

  cin >> n;

  /**
   * Block Logic: This loop processes each of the 'n' vectors in a single pass.
   */
  while (n--) {
    // Read the components of the current vector.
    cin >> x >> y >> z;
    // Add the components to their respective running sums immediately.
    xsum += x;
    ysum += y;
    zsum += z;
  }

  /**
   * Decision Logic: The system is in equilibrium if and only if the net force
   * along each axis is zero.
   */
  if (xsum == 0 && ysum == 0 && zsum == 0) {
    cout << "YES" << endl;
  } else
    cout << "NO" << endl;

  return 0;
}
