/**
 * @file 69A.CPP
 * @brief Solution to Codeforces problem 69A - Young Physicist.
 *        This program determines if a system of forces acting on a body is in equilibrium.
 *        Equilibrium is achieved if the sum of all force components along the X, Y, and Z axes is zero.
 * Algorithm:
 *   1. Read the number of force vectors (n).
 *   2. Initialize sum variables for x, y, and z components (f1, f2, f3) to zero.
 *   3. Loop 'n' times:
 *      a. Read the three components (t1, t2, t3) of each force vector.
 *      b. Add each component to its respective sum (f1, f2, f3).
 *   4. After processing all force vectors, check if f1, f2, and f3 are all zero.
 *   5. Print "YES" if all sums are zero (system is in equilibrium), otherwise print "NO".
 * Time Complexity: O(n), where n is the number of force vectors. Each vector requires constant time processing.
 * Space Complexity: O(1) as only a few variables are used to store sums.
 */
#include <bits/stdc++.h>
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
  int n;
  cin >> n;
  int t1, t2, t3, f1, f2, f3;
  f1 = f2 = f3 = 0;
  // Block Logic: Iterates 'n' times to read each force vector and accumulate its components.
  // Pre-condition: 'n' specifies the total number of force vectors to process.
  // Invariant: 'f1', 'f2', and 'f3' maintain the running totals of force components.
  for (int i = 0; i < n; i++) {
    cin >> t1 >> t2 >> t3;
    f1 += t1;
    f2 += t2;
    f3 += t3;
  }
  // Block Logic: Checks if the system of forces is in equilibrium.
  // Pre-condition: 'f1', 'f2', and 'f3' hold the final resultant force components.
  // Post-condition: Prints "YES" if all components are zero (equilibrium), otherwise "NO".
  if (f1 || f2 || f3) { // Checks if any of the sums are non-zero.
    cout << "NO" << endl;
  } else {
    cout << "YES" << endl;
  }
  return 0;
}
