/**
 * @file 69A.CPP
 * @brief Solves the "Young Physicist" problem by checking for force equilibrium.
 * @details This program determines if a body is in equilibrium under the influence of
 * multiple forces. According to Newton's First Law, a body is in equilibrium if the
 * vector sum of all forces acting on it is zero. This program checks that condition.
 *
 * @algorithm The program reads `n` 3D force vectors. It maintains a running sum for
 * each of the x, y, and z components. After summing all the vectors, it checks if the
 * resultant vector's components are all zero. If and only if the sum of components for
 * all three axes is zero, the body is in equilibrium.
 *
 * @time_complexity O(N), where N is the number of force vectors, due to a single pass
 * to read and sum the vectors.
 * @space_complexity O(1), as a fixed number of variables are used to store the component sums.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  // n: The number of force vectors to be read.
  // x, y, z: Components of a single force vector.
  // sx, sy, sz: State variables for the sum of components of the resultant vector.
  int n, x, y, z, sx = 0, sy = 0, sz = 0;
  cin >> n;
  /**
   * Block Logic: Iterate through all `n` force vectors, accumulating the components.
   * Invariant: At the end of each iteration, `sx`, `sy`, and `sz` hold the component
   *            sums of all vectors processed so far.
   */
  for (int i = 0; i < n; i++) {
    cin >> x >> y >> z;
    sx += x, sy += y, sz += z;
  }
  /**
   * Block Logic: Check for equilibrium.
   * Pre-condition: `sx`, `sy`, and `sz` represent the components of the net force vector.
   * A body is in equilibrium if and only if the net force is the zero vector.
   */
  if (sx == 0 && sy == 0 && sz == 0)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}
