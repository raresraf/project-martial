/**
 * @file 69A.CPP
 * @brief Solution for Codeforces Problem 69A - "Young Physicist".
 * @details The problem determines if a body is in equilibrium under the influence of multiple forces.
 * A body is in equilibrium if the vector sum of all forces acting on it is the zero vector.
 * This means the sum of the x-components, y-components, and z-components of all forces must each be zero.
 *
 * @author (unknown)
 *
 * @algorithm
 * 1. Read the total number of forces, `n`.
 * 2. Initialize a vector `a` of size 3 to all zeros. This vector will accumulate the
 *    sum of force components for the x, y, and z axes, respectively.
 * 3. Loop `n` times, once for each force vector.
 * 4. In each iteration, read the x, y, and z components of the current force.
 * 5. Add each component to the corresponding element in the sum vector `a`.
 * 6. After the loop, the vector `a` holds the resultant force vector.
 * 7. Compare the resultant vector `a` with a zero vector. If they are equal
 *    (i.e., `a` is `[0, 0, 0]`), the body is in equilibrium. Print "YES".
 * 8. Otherwise, print "NO".
 *
 * @complexity
 * Time Complexity: O(n), where `n` is the number of forces, due to the single loop for summation.
 * Space Complexity: O(1), as the storage used (two vectors of size 3) is constant.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  cin >> n;
  // `a` will store the sum of force components [x, y, z].
  // `b` is a zero vector for comparison.
  vector<int> a(3, 0), b(3, 0);
  /**
   * Block Logic: This loop reads each force vector and accumulates the component sums.
   */
  while (n--) {
    int q, w, e; // x, y, and z components of a single force.
    cin >> q >> w >> e;
    // Add the components to the total sum vector.
    a[0] += q; // Sum of x-components
    a[1] += w; // Sum of y-components
    a[2] += e; // Sum of z-components
  }
  /**
   * Block Logic: Checks for equilibrium.
   * The object is in equilibrium if the resultant force vector `a` is the zero vector.
   * The comparison `a == b` checks if all corresponding elements in the vectors are equal.
   */
  if (a == b)
    cout << "YES";
  else
    cout << "NO";
}