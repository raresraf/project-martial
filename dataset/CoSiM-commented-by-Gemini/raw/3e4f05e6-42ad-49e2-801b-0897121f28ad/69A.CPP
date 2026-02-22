/**
 * @file 69A.CPP
 * @brief Solution for the "Young Physicist" problem (Codeforces 69A).
 * @see http://codeforces.com/problemset/problem/69/A
 *
 * @algorithm
 * The problem is to determine if a body is in equilibrium, which occurs when the
 * vector sum of all forces acting on it is zero. This requires the sum of the
 * components along each axis (x, y, and z) to be zero independently.
 *
 * This solution calculates the sum of components on the fly:
 * 1. Read the number of force vectors, `n`.
 * 2. Initialize sum variables for each axis (`xf`, `yf`, `zf`) to 0.
 * 3. Loop `n` times. In each iteration, read the (x, y, z) components of a
 *    force vector.
 * 4. Add each component to its corresponding sum variable.
 * 5. After the loop, check if all three sum variables are equal to 0.
 * 6. If they are, the body is in equilibrium ("YES"); otherwise, it is not ("NO").
 *
 * @complexity
 * - Time Complexity: O(N), for the single pass through the N force vectors.
 * - Space Complexity: O(N), due to the use of Variable Length Arrays (VLAs) to
 *   store all vector components. Note that this is not space-optimal, as the
 *   problem can be solved with O(1) space by not storing the arrays.
 */
#include <iostream>
using namespace std;

int main() {
  int n;
  cin >> n;

  // Use of Variable Length Arrays (VLAs), a non-standard C++ extension.
  // These arrays are not strictly necessary as the sums are computed on the fly.
  int x[n], y[n], z[n];
  // `xf`, `yf`, `zf` will store the sum of forces along each axis.
  int xf = 0, yf = 0, zf = 0;
  /**
   * @brief Loop to read each force vector and add its components to the sums.
   */
  for (int i = 0; i < n; ++i) {
    cin >> x[i] >> y[i] >> z[i];
    xf += x[i];
    yf += y[i];
    zf += z[i];
  }

  /**
   * @brief Check for equilibrium. The body is in equilibrium if the net force
   * along each axis is zero.
   */
  if (xf == 0 && yf == 0 && zf == 0)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
/*
http://codeforces.com/problemset/problem/69/A
*/
