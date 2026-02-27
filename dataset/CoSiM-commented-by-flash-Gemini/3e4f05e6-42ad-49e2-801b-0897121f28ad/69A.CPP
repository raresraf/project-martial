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
  // Declare 'n' for the number of forces.
  int n;
  // Read the number of forces from standard input.
  cin >> n;

  // Declare arrays to store the x, y, and z components of each force.
  // Note: Using VLA (Variable Length Array) x[n], y[n], z[n] which is a C99 feature
  // and might not be supported by all C++ compilers. For competitive programming,
  // it's often accepted or specific compiler extensions are used.
  int x[n], y[n], z[n];
  // Initialize accumulators for the sum of x, y, and z components.
  int xf = 0, yf = 0, zf = 0;
  
  // Block Logic: Read each force's components and accumulate their sums.
  // Invariant: 'xf', 'yf', 'zf' store the running sum of x, y, z components respectively.
  for (int i = 0; i < n; ++i) {
    // Read the x, y, and z components of the current force.
    cin >> x[i] >> y[i] >> z[i];
    // Accumulate sums.
    xf += x[i];
    yf += y[i];
    zf += z[i];
  }

  // Block Logic: Determine if the body is in equilibrium.
  // Precondition: A body is in equilibrium if all three component sums are zero.
  if (xf == 0 && yf == 0 && zf == 0)
    cout << "YES" << endl; // Output "YES" if in equilibrium.
  else
    cout << "NO" << endl;  // Output "NO" otherwise.

  return 0; // Indicate successful program termination.
}
/*
http://codeforces.com/problemset/problem/69/A
*/
