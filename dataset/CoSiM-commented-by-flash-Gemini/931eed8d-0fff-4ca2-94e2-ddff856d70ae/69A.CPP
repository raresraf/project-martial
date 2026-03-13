/**
 * @file 69A.CPP
 * @brief Solves the "Young Physicist" problem from Codeforces (69A).
 * @details This program determines if a body is in equilibrium by calculating the
 * net force from a given set of vectors. This implementation is memory-efficient.
 *
 * @section Algorithm
 * The principle of equilibrium states that the vector sum of all forces acting on
 * a body must be zero. This means the sum of the x, y, and z components must
 * each be zero.
 *
 * The algorithm implements this check efficiently:
 * 1. It reads 'n', the total number of force vectors.
 * 2. It iterates 'n' times in a single loop. In each iteration, it reads the
 *    (x, y, z) components of a vector.
 * 3. Instead of storing each vector, it immediately adds the components to running
 *    totals for each axis (`xt`, `yt`, `zt`).
 * 4. After the loop, it checks if all three running totals are zero. If so, the
 *    net force is zero, and the body is in equilibrium ("YES"). Otherwise,
 *    it is not ("NO").
 *
 * @section Complexity
 * - Time Complexity: O(n), for a single pass through the 'n' force vectors.
 * - Space Complexity: O(1), as it only requires a few variables for the running
 *   totals, regardless of the number of vectors.
 */
#include <iostream>
using namespace std;

int main() {
  /**
   * @var n The total number of force vectors to process.
   * @var i Loop counter.
   * @var x, y, z Temporary storage for the components of a single input vector.
   * @var xt, yt, zt Running totals for the sum of components along each axis.
   */
  int n, i;
  int x, y, z;
  int xt = 0, yt = 0, zt = 0;

  // Read the number of force vectors.
  cin >> n;

  /**
   * @brief Block Logic: Iterates through each force vector, reads its components,
   * and adds them to the running totals. This combines input and processing
   * into a single, memory-efficient loop.
   */
  for (i = 0; i < n; i++) {
    cin >> x >> y >> z;
    xt = xt + x;
    yt = yt + y;
    zt = zt + z;
  }

  /**
   * @brief Logic: Checks for equilibrium. The condition is met if the net force
   * along each of the three axes is zero.
   */
  if (xt == 0 && yt == 0 && zt == 0) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  return 0;
}
