/**
 * @file 69A.CPP
 * @brief Solution for competitive programming problem: "Young Physicist".
 *
 * @details This program determines if a body is in equilibrium given a set of 'n'
 *          forces acting on it. A body is in equilibrium if the sum of the
 *          x-components, y-components, and z-components of all forces is zero.
 *          Algorithm: Summation of corresponding vector components.
 *          Time Complexity: O(N), where N is the number of forces, as each force's
 *                           components are read and added once.
 *          Domain: Competitive Programming.
 */
#include <iostream>
using namespace std;

int main() {
  // Declare variables:
  // 'n': number of forces.
  // 'x', 'y', 'z': components of the current force.
  // 'xs', 'ys', 'zs': accumulators for the sum of x, y, z components across all forces.
  int n, x, y, z, xs = 0, ys = 0, zs = 0;
  // Read the number of forces from standard input.
  cin >> n;
  
  // Block Logic: Iterate 'n' times, reading each force's components and accumulating their sums.
  // Invariant: 'xs', 'ys', 'zs' store the running sum of x, y, z components respectively.
  while (n--) { // Loop 'n' times.
    // Read the x, y, and z components of the current force.
    cin >> x >> y >> z;
    // Accumulate sums for each component.
    xs += x;
    ys += y;
    zs += z;
  }
  
  // Block Logic: Determine if the body is in equilibrium.
  // Precondition: A body is in equilibrium if the sum of all components in each dimension is zero.
  if (xs == 0 && ys == 0 && zs == 0) {
    cout << "YES"; // Output "YES" if in equilibrium.
  } else {
    cout << "NO";  // Output "NO" otherwise.
  }
}