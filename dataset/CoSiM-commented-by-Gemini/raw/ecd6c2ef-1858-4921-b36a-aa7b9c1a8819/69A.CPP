/**
 * @file 69A.CPP
 * @brief Solution to a competitive programming problem checking for equilibrium of forces.
 * This program reads a series of 3D force vectors and determines if the sum of all
 * components (x, y, and z) individually equals zero, indicating a state of equilibrium.
 *
 * Algorithm: Summation of vector components.
 * Time Complexity: O(N) where N is the number of force vectors.
 * Space Complexity: O(1) for storing sum variables.
 */
#include <iostream>
using namespace std;
/**
 * @brief Main function of the program.
 * Reads the number of force vectors `n`, then processes `n` vectors to calculate
 * the sum of their x, y, and z components. Finally, it determines if the object
 * is in equilibrium based on these sums.
 * @return `0` upon successful execution.
 */
int main() {
  int n, x, y, z, xsum(0), ysum(0), zsum(0);
  cin >> n;
  /**
   * @brief Loop to read and accumulate components of force vectors.
   * Invariant: `n` decrements in each iteration, reading one vector (x, y, z)
   * and adding its components to `xsum`, `ysum`, and `zsum` respectively.
   */
  while (n--) {
    cin >> x >> y >> z;
    xsum += x;
    ysum += y;
    zsum += z;
  }
  /**
   * @brief Checks for equilibrium condition.
   * If the sum of all x, y, and z components are all zero, the object is in
   * equilibrium ("YES"). Otherwise, it is not ("NO").
   */
  if (xsum == 0 && ysum == 0 && zsum == 0) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
  return 0;
}