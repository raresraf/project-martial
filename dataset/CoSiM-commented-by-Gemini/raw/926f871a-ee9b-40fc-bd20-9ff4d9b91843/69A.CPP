/**
 * @file 69A.CPP
 * @brief Determines if a system of 3D vectors is in equilibrium.
 *
 * This program reads a set of 3D vectors and checks if their vector sum is zero.
 * This is a classic problem in physics, representing a check for static equilibrium
 * where the net force on an object is zero. If the sum of all x, y, and z components
 * is zero, the system is in equilibrium.
 *
 * @algorithm Vector Summation
 * @complexity
 * - Time: O(N), where N is the number of vectors, because it iterates through the
 *   input vectors once.
 * - Space: O(1), as it only uses a few variables to store the running sums,
 *   independent of the number of vectors.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  // n: number of vectors. x, y, z: components of each input vector.
  int n, x, y, z;
  // x1, y1, z1: store the sum of the x, y, and z components respectively.
  int x1 = 0;
  int y1 = 0;
  int z1 = 0;
  // The 'total' and 'sum' variables are declared but never used.
  int total;
  int sum = 0;
  cin >> n;

  /**
   * @brief Block Logic: Reads each vector and accumulates the component sums.
   * Invariant: After each iteration `i`, `x1`, `y1`, and `z1` hold the sum of
   *            components for the first `i` vectors.
   */
  for (int i = 1; i <= n; i++) {
    cin >> x >> y >> z;
    x1 = x1 + x;
    y1 = y1 + y;
    z1 = z1 + z;
  }

  /**
   * @brief Block Logic: Checks if the net force vector is zero.
   * Pre-condition: The sums of all x, y, and z components have been calculated.
   */
  if (x1 == 0 && y1 == 0 && z1 == 0) {
    // If the net force is zero, the system is in equilibrium.
    cout << "YES";
  } else {
    // Otherwise, it is not in equilibrium.
    cout << "NO";
  }
}