/**
 * @file 69A.CPP
 * @brief Solution for the "Young Physicist" problem (Codeforces 69A).
 *
 * @algorithm
 * The problem asks to determine if a body is in equilibrium given `m` force
 * vectors acting upon it. A body is in equilibrium if and only if the vector
 * sum of all forces is zero. This means the sum of the x-components, the
 * y-components, and the z-components must all independently be zero.
 *
 * The algorithm is as follows:
 * 1. Initialize three variables `sum_x`, `sum_y`, `sum_z` to 0.
 * 2. Read the number of force vectors, `m`.
 * 3. Loop `m` times. In each iteration, read a force vector (x, y, z).
 * 4. Add the components of the read vector to the corresponding sum variables.
 * 5. After the loop, check if `sum_x`, `sum_y`, and `sum_z` are all equal to 0.
 * 6. If they are, the body is in equilibrium ("YES"); otherwise, it is not ("NO").
 *
 * @complexity
 * - Time Complexity: O(M), where M is the number of force vectors, due to the
 *   single pass to sum the vectors.
 * - Space Complexity: O(1), as the storage required is constant.
 */
#include <bits/stdc++.h>
using namespace std;
// This typedef is declared but never used in the program.
typedef long long ll;

int main() {
  // x, y, z: components of a single force vector.
  // a, b, c: running sums for the x, y, and z components, respectively.
  // m: the number of force vectors.
  // n, i, j, k: unused variables.
  int x, y, z, a = 0, b = 0, c = 0, m, n, i, j, k;
  cin >> m;
  /**
   * @brief Loop `m` times to read each force vector and add its components
   * to the total sums.
   */
  while (m--) {
    cin >> x >> y >> z;
    a += x, b += y, c += z;
  }
  /**
   * @brief Check for equilibrium. The net force is zero if and only if the
   * sum of components along each axis is zero.
   */
  if (a == 0 && b == 0 && c == 0)
    cout << "YES" << endl;
  else
    puts("NO"); // puts is a C-style function to print a string with a newline.

  return 0;
}
