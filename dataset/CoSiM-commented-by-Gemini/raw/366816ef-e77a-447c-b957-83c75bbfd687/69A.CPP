/**
 * @file 69A.CPP
 * @brief Determines if a set of 3D vectors sums to the zero vector.
 *
 * This program solves the "Young Physicist" problem. It checks if a body
 * is in equilibrium by summing up 'n' force vectors. If the resultant
 * vector is (0, 0, 0), the body is in equilibrium.
 *
 * Algorithm:
 * 1. Read the number of vectors, n.
 * 2. Initialize three variables to store the sum of the x, y, and z
 *    components of the vectors.
 * 3. Loop 'n' times, reading each vector and adding its components to the
 *    corresponding sum variables.
 * 4. After the loop, check if all three sums are zero.
 * 5. Print "YES" if they are, "NO" otherwise.
 *
 * Time Complexity: O(N), where N is the number of vectors.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  // i: loop counter.
  // n: the number of force vectors.
  // x, y, z: components of a single vector.
  // sumx, sumy, sumz: accumulators for the total force on each axis.
  int i = 0, n, x, y, z, sumx = 0, sumy = 0, sumz = 0;
  
  cin >> n; // Read the number of vectors.

  /**
   * Block Logic: Summation of all force vectors.
   * This loop iterates 'n' times to read each vector and add its
   * components to the running totals.
   */
  while (i < n) {
    cin >> x >> y >> z;
    sumx += x;
    sumy += y;
    sumz += z;
    i++;
  }
  
  /**
   * Block Logic: Equilibrium check.
   * The condition for equilibrium is that the net force along each
   * axis is zero.
   */
  if (sumx == 0 && sumy == 0 && sumz == 0)
    cout << "YES";
  else
    cout << "NO";
    
  return 0;
}
