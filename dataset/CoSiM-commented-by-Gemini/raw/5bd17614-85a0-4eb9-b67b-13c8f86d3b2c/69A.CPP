/**
 * @file 69A.CPP
 * @brief Solution for a vector equilibrium problem.
 * @details This program determines if a body is in equilibrium under the action of
 *          multiple forces in 3D space. Equilibrium is achieved if the vector
 *          sum of all forces is zero.
 *
 * Algorithm:
 * 1. Read the number of force vectors, 'n'.
 * 2. Initialize three variables (sum1, sum2, sum3) to 0 to store the sum of
 *    the x, y, and z components of the vectors, respectively.
 * 3. Loop 'n' times:
 *    a. Read the x, y, and z components of a force vector.
 *    b. Add each component to its corresponding sum.
 * 4. After the loop, check if all three sums are zero.
 * 5. If sum1, sum2, and sum3 are all 0, the body is in equilibrium, so print "YES".
 * 6. Otherwise, print "NO".
 *
 * Time Complexity: O(n), where 'n' is the number of vectors.
 * Space Complexity: O(n) due to the storage of vector components in arrays,
 *                   although it could be O(1) if components were processed
 *                   without storage.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  // n: number of force vectors.
  // sum1, sum2, sum3: store the sum of x, y, and z components respectively.
  int n, sum1 = 0, sum2 = 0, sum3 = 0;
  cin >> n;
  // x, y, z: arrays to store the components of the n vectors.
  // Note: Using Variable Length Arrays (VLA) which is a non-standard C++ extension.
  int x[n], y[n], z[n];

  /**
   * Block Logic: Reads each vector's components and accumulates them into the
   * respective sum variables.
   * Invariant: At the end of each iteration, sum1, sum2, and sum3 contain the
   * component-wise sum of all vectors read so far.
   */
  for (int i = 0; i < n; i++) {
    cin >> x[i] >> y[i] >> z[i];
    sum1 += x[i];
    sum2 += y[i];
    sum3 += z[i];
  }

  /**
   * Block Logic: Checks for the equilibrium condition.
   * A body is in equilibrium if and only if the net force acting on it is zero.
   * This translates to the sum of each component of the force vectors being zero.
   */
  if (sum1 == 0 && sum2 == 0 && sum3 == 0)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}