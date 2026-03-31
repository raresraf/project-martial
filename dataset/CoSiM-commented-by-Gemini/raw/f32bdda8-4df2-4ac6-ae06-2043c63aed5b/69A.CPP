/**
 * @file 69A.CPP
 * @brief Solves the "Young Physicist" problem by checking for vector equilibrium.
 *
 * This program determines if a body is in equilibrium under the influence of
 * several force vectors. According to Newton's first law, a body is in
 * equilibrium if the vector sum of all forces acting on it is zero. This means
 * the sum of the x-components, the sum of the y-components, and the sum of the
 * z-components of all force vectors must each be zero. The program reads the
 * components of `n` vectors and checks this condition.
 *
 * @author f32bdda8-4df2-4ac6-ae06-2043c63aed5b
 *
 * @algorithm Vector Component Summation
 * @complexity
 * Time: O(N), where N is the number of force vectors, due to the single loop.
 * Space: O(1), as the memory usage is constant.
 */
#include <iostream>
using namespace std;
int main() {
  // `n` is the number of force vectors.
  int n;
  cin >> n;
  // `xsum`, `ysum`, `zsum` will accumulate the sum of components for each axis.
  int xsum = 0, ysum = 0, zsum = 0;

  /**
   * @brief Loop to read each force vector and update the component sums.
   */
  while (n--) {
    // `x`, `y`, `z` are the components of a single force vector.
    int x, y, z;
    cin >> x >> y >> z;
    // Add the components to their respective sums.
    xsum += x;
    ysum += y;
    zsum += z;
  }

  /**
   * @brief Check for equilibrium condition.
   * The body is in equilibrium if and only if the sum of forces along each
   * of the three axes is zero.
   */
  if (xsum == 0 && ysum == 0 && zsum == 0) {
    cout << "YES" << '
';
  } else {
    cout << "NO" << '
';
  }
}