/**
 * @file 69A.CPP
 * @brief Solves the "Young Physicist" problem by checking for vector equilibrium.
 *
 * This program determines if a system of 3D forces is in equilibrium. It reads
 * `n` force vectors and checks if their vector sum is the zero vector. This is
 * true if and only if the sum of all x, y, and z components are each zero.
 *
 * @algorithm
 * This implementation uses an efficient, single-pass approach.
 *
 * 1. Read the total number of vectors, `n`.
 * 2. Initialize three sum variables, `f1`, `f2`, and `f3`, to 0. These will
 *    store the net force components along the x, y, and z axes, respectively.
 * 3. Loop `n` times to process each vector.
 *    a. In each iteration, read the three components of a vector into temporary
 *       variables `t1`, `t2`, and `t3`.
 *    b. Add these components to their respective running sums (`f1`, `f2`, `f3`).
 * 4. After the loop, `f1`, `f2`, and `f3` represent the components of the resultant
 *    force vector.
 * 5. Check if the resultant force is non-zero. The condition `if (f1 || f2 || f3)`
 *    leverages C++'s behavior where any non-zero integer evaluates to `true`.
 *    The condition is true if any of the component sums is not zero.
 *    a. If true, the system is not in equilibrium; print "NO".
 *    b. If false (meaning all component sums are zero), the system is in
 *       equilibrium; print "YES".
 *
 * @complexity
 * Time Complexity: O(n), due to the single loop over the `n` vectors.
 * Space Complexity: O(1), as it only uses a few variables to store the running
 * sums, regardless of the number of vectors. This is a very memory-efficient
 * approach.
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  cin >> n;

  // t1, t2, t3: temporary variables for the components of the current vector.
  // f1, f2, f3: store the sum of forces (components) along x, y, and z axes.
  int t1, t2, t3, f1, f2, f3;
  
  // Initialize the sum of forces to zero.
  f1 = f2 = f3 = 0;

  /**
   * @brief Process each vector, adding its components to the total sums.
   */
  for (int i = 0; i < n; i++) {
    cin >> t1 >> t2 >> t3;
    f1 += t1;
    f2 += t2;
    f3 += t3;
  }

  // Check for equilibrium. `f1 || f2 || f3` is true if any component is non-zero.
  if (f1 || f2 || f3) {
    // If there is a net force in any direction, it's not in equilibrium.
    cout << "NO" << endl;
  } else {
    // If all force components sum to zero, it is in equilibrium.
    cout << "YES" << endl;
  }

  return 0;
}
