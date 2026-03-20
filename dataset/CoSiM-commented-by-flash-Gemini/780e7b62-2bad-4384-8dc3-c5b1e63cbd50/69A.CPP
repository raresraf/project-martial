/**
 * @file 69A.CPP
 * @brief Solution to Codeforces problem 69A - Young Physicist.
 *        This program determines if a system of forces acting on a body is in equilibrium.
 *        Equilibrium is achieved if the sum of all force components along the X, Y, and Z axes is zero.
 * Algorithm:
 *   1. Read the number of force vectors (n).
 *   2. Initialize sum variables for x, y, and z components to zero.
 *   3. Loop 'n' times:
 *      a. Read the three components (x, y, z) of each force vector.
 *      b. Add each component to its respective sum (xsum, ysum, zsum).
 *   4. After processing all force vectors, check if xsum, ysum, and zsum are all zero.
 *   5. Print "YES" if all sums are zero (system is in equilibrium), otherwise print "NO".
 * Time Complexity: O(n), where n is the number of force vectors. Each vector requires constant time processing.
 * Space Complexity: O(n) for storing the force components, although O(1) would be possible by processing sums directly.
 */
#include <iostream>
using namespace std;
int main() {
  /**
   * @brief Main function to solve the "Young Physicist" problem.
   *
   * Reads the number of force vectors and then iterates through each vector,
   * summing its x, y, and z components. Finally, it checks if all three
   * resultant sums are zero to determine if the body is in equilibrium.
   * @return int Returns 0 upon successful execution.
   */

  int x[101], y[101], z[101];
  int n;
  int xsum = 0, ysum = 0, zsum = 0;
  cin >> n;
  // Block Logic: Iterates 'n' times to read each force vector and accumulate its components.
  // Pre-condition: 'n' specifies the total number of force vectors to process.
  // Invariant: 'xsum', 'ysum', 'zsum' maintain the running totals of components.
  for (int i = 0; i < n; i++) {
    cin >> x[i] >> y[i] >> z[i];
    xsum += x[i];
    ysum += y[i];
    zsum += z[i];
  }
  // Block Logic: Checks if the system of forces is in equilibrium.
  // Pre-condition: 'xsum', 'ysum', 'zsum' hold the final resultant force components.
  // Post-condition: Prints "YES" if all components are zero (equilibrium), otherwise "NO".
  if (xsum == 0 && ysum == 0 && zsum == 0)
    cout << "YES";
  else
    cout << "NO";
  return 0;
}
