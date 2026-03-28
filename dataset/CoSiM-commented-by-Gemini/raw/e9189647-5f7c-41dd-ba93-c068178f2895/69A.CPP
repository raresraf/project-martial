/**
 * @file 69A.CPP
 * @brief Solves the "Young Physicist" problem from Codeforces.
 *
 * This program determines if a body is in equilibrium by checking if the vector
 * sum of all forces acting upon it is zero. The forces are given as 3D vectors.
 *
 * @algorithm The program reads `n` force vectors. It maintains three variables
 * to store the sum of all x, y, and z components respectively. It iterates
 * through the `n` vectors, accumulating the components. After processing all
 * vectors, it checks if the sums of all three components are zero. If and only
 * if all three sums are zero, the body is in equilibrium.
 *
 * @time_complexity O(N), where N is the number of force vectors.
 * @space_complexity O(1).
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, y, x, z, suma = 0, sumb = 0, sumc = 0;
  cin >> n;
  /**
   * @block_logic This loop reads each of the `n` vectors and accumulates the
   * x, y, and z components into `suma`, `sumb`, and `sumc` respectively.
   */
  while (n--) {
    cin >> x >> y >> z;
    suma = suma + x;
    sumb = sumb + y;
    sumc = sumc + z;
  }
  /**
   * @block_logic Checks if the resultant force vector is the zero vector.
   * In C++, a non-zero integer evaluates to true. The logical OR `||` checks
   * if any component of the sum is non-zero. If any is non-zero, the body is
   * not in equilibrium.
   */
  if (suma || sumb || sumc)
    cout << "NO" << endl;
  else
    cout << "YES" << endl;
  return 0;
}
