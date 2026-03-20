/**
 * @file 69A.CPP
 * @brief Solution for Codeforces problem 69A, "Young Physicist".
 * @details This program checks if a body is in equilibrium by summing the components
 * of all force vectors acting on it and determining if the net force is zero.
 *
 * @section Algorithm
 * According to physics, a body is in equilibrium if the vector sum of all forces
 * is the zero vector. This means the sum of the x-components, the sum of the
 * y-components, and the sum of the z-components must all be zero.
 *
 * 1. The program reads an integer `many`, the number of force vectors.
 * 2. It initializes three accumulators, `xt`, `yt`, and `zt`, to zero.
 * 3. It loops `many` times, reading the x, y, and z components of each vector.
 * 4. Each component is added to its corresponding accumulator.
 * 5. After the loop, it checks if all three accumulators are zero. The condition
 *    `xt == yt && xt == zt && xt == 0` is a verbose but correct way to verify this.
 * 6. It prints "YES" if the body is in equilibrium and "NO" otherwise.
 *
 * @subsection Complexity
 * - Time Complexity: O(n), where `n` is the number of vectors, due to the single loop.
 * - Space Complexity: O(1), as memory usage is constant.
 */
/**
 * I'm sorry I was an idiot
 */
#include <iostream>

using namespace std;

int main() {
  // `many`: stores the number of force vectors.
  int many;
  cin >> many;
  // `xt`, `yt`, `zt`: accumulators for the sum of forces on the x, y, and z axes.
  int xt = 0, yt = 0, zt = 0;
  /**
   * @brief This loop reads the components of each force vector and adds them
   * to the total sum for each axis.
   */
  while (many--) {
    int x, y, z;
    cin >> x >> y >> z;
    xt += x;
    yt += y;
    zt += z;
  }
  /**
   * @brief This condition checks for equilibrium. A body is in equilibrium if the
   * net force on it is zero, meaning the sum of forces along each axis is zero.
   * The check `xt == yt && xt == zt && xt == 0` is a logically sound, if verbose,
   * way to confirm that all three sums are zero.
   */
  if (xt == yt && xt == zt && xt == 0) {
    cout << "YES" << endl;
    return 0;
  }
  cout << "NO" << endl;
  return 0;
}