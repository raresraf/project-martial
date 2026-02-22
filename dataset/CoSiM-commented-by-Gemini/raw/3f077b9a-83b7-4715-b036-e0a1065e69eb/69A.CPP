/**
 * @file 69A.CPP
 * @brief Solution for the "Young Physicist" problem (Codeforces 69A).
 *
 * @algorithm
 * The problem asks to determine if a body is in equilibrium given `n` force
 * vectors acting upon it. A body is in equilibrium if the vector sum of all
 * forces is zero. This requires the sum of the components along each axis
 * (x, y, and z) to be zero independently.
 *
 * This solution efficiently calculates the net force by summing the components
 * without storing the individual vectors:
 * 1. Initialize three sum variables (`xs`, `ys`, `zs`) to 0.
 * 2. Read the number of force vectors, `n`.
 * 3. Loop `n` times. In each iteration, read a force vector's (x, y, z) components.
 * 4. Add each component to its corresponding sum variable.
 * 5. After the loop, if all three sums are 0, the body is in equilibrium ("YES").
 *    Otherwise, it is not ("NO").
 *
 * @complexity
 * - Time Complexity: O(N), for a single pass through the N force vectors.
 * - Space Complexity: O(1), as only a few variables are needed to store the sums.
 */
#include <iostream>
using namespace std;

int main() {
  int n;
  // x, y, z: components of a single force vector.
  // xs, ys, zs: running sums for the x, y, and z components.
  int x, y, z, xs = 0, ys = 0, zs = 0;
  cin >> n;
  /**
   * @brief Loop `n` times to read each force vector and add its components
   * to the total sums.
   */
  while (n--) {
    cin >> x >> y >> z;
    xs += x;
    ys += y;
    zs += z;
  }
  /**
   * @brief Check for equilibrium. The body is in equilibrium if the net force
   * along each axis is zero.
   */
  if (xs == 0 && ys == 0 && zs == 0) {
    cout << "YES";
  } else {
    cout << "NO";
  }
}