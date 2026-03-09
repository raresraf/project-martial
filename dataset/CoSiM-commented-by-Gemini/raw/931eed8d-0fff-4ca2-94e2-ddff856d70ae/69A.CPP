/**
 * @file 69A.CPP
 * @brief Determines if a system of 3D vectors is in equilibrium.
 *
 * This program reads a specified number of 3D vectors and determines if their
 * vector sum is the zero vector. This is analogous to checking for static
 * equilibrium in a physical system, where the net force is zero.
 *
 * @algorithm Vector Summation
 * @complexity
 * - Time: O(N), where N is the number of vectors, as the input is processed in a single loop.
 * - Space: O(1), as memory usage is constant regardless of the number of vectors.
 */
#include <iostream>
using namespace std;

int main() {
  // n: the number of vectors to be read.
  // i: loop counter.
  int n, i;
  // x, y, z: components of each individual vector.
  int x, y, z;
  // xt, yt, zt: running totals for the x, y, and z components of the net vector.
  int xt = 0, yt = 0, zt = 0;
  cin >> n;

  /**
   * @brief Block Logic: Reads `n` vectors and accumulates their components.
   * Invariant: After each iteration, `xt`, `yt`, and `zt` hold the vector sum
   *            of all vectors read so far.
   */
  for (i = 0; i < n; i++) {
    cin >> x >> y >> z;
    xt = xt + x;
    yt = yt + y;
    zt = zt + z;
  }

  /**
   * @brief Block Logic: Checks if the total vector sum is zero.
   * Pre-condition: All vectors have been read and their components summed.
   */
  if (xt == 0 && yt == 0 && zt == 0) {
    // If the net vector is (0,0,0), the system is in equilibrium.
    cout << "YES";
  } else {
    // Otherwise, there is a non-zero net force.
    cout << "NO";
  }
  return 0;
}
