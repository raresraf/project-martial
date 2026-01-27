/**
 * @file 69A.CPP
 * @brief Solves the "Young Physicist" problem from Codeforces.
 * @details This program determines if a body is in equilibrium by checking if the
 * net force acting on it is zero. It reads a series of 3D force vectors and
 * calculates the sum of their components along the x, y, and z axes. If all
 * three component sums are zero, the body is in equilibrium.
 *
 * @see https://codeforces.com/problemset/problem/69/A
 *
 * Algorithm:
 * 1. Read the number of force vectors, `n`.
 * 2. Initialize three variables `t1`, `t2`, `t3` to store the sum of forces
 *    along the x, y, and z axes, respectively.
 * 3. Loop `n` times to read each force vector `(x, y, z)`.
 * 4. In each iteration, add the x, y, and z components to their respective sums.
 * 5. After the loop, check if `t1`, `t2`, and `t3` are all zero.
 * 6. If all sums are zero, print "YES". Otherwise, print "NO".
 *
 * Time Complexity: O(n), as the program iterates through the `n` vectors once.
 * Space Complexity: O(1), as it uses a fixed amount of memory.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // n: the number of force vectors.
  int n;
  cin >> n;
  // x, y, z: components of a single force vector.
  int x, y, z;
  // t1, t2, t3: sum of forces along the x, y, and z axes respectively.
  int t1 = 0, t2 = 0, t3 = 0;

  /**
   * Block Logic: Read n force vectors and compute the sum of their components.
   * Invariant: At the end of each iteration, t1, t2, and t3 hold the cumulative
   * sum of the components of the vectors read so far.
   */
  for (int i = 0; i < n; i++) {
    cin >> x >> y >> z;
    t1 += x;
    t2 += y;
    t3 += z;
  }

  /**
   * Block Logic: Check for equilibrium.
   * The condition for equilibrium is that the net force vector is zero, which means
   * the sum of components along each axis must be zero.
   */
  if (t1 == 0 && t2 == 0 && t3 == 0) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
  return 0;
}