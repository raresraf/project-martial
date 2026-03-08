/**
 * @file 69A.CPP
 * @brief Solution for the "Young Physicist" vector equilibrium problem (e.g., Codeforces 69A).
 * @author Sagindyk Urazayev
 * @details This program determines if a body is in equilibrium by summing the x, y, and z
 * components of all applied force vectors. If the resultant vector is (0, 0, 0),
 * the body is in equilibrium.
 *
 * @algorithm Vector Summation
 * @complexity
 * - Time: O(n), where 'n' is the number of force vectors, as the program iterates
 *   through each vector once.
 * - Space: O(1), using a fixed number of variables for the sums.
 */

#include <iostream>

using namespace std;

int main() {
  // zres, yres, xres: store the sum of forces along the z, y, and x axes.
  // n: the number of force vectors.
  // i: loop counter.
  // x, y, z: components of a single force vector.
  int zres, yres, xres, n, i, x, y, z;
  
  // Initialize the sums of the vector components to zero.
  xres = yres = zres = 0;
  
  cin >> n;

  /**
   * @block Vector Summation Loop
   * @logic Reads each of the 'n' force vectors and adds their components to the
   * corresponding running total for each axis.
   */
  for (i = 0; i < n; i++) {
    cin >> x >> y >> z;
    xres = xres + x;
    yres = yres + y;
    zres = zres + z;
  }

  /**
   * @block Equilibrium Check
   * @logic If the sum of forces along all three axes is zero, the body is in equilibrium.
   */
  if ((xres == 0) && (yres == 0) && (zres == 0)) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
  
  return 0;
}
