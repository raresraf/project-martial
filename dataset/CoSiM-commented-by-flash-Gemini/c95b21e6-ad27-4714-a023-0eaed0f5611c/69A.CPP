/**
 * @file 69A.CPP
 * @brief This program determines if a system of 'n' 3D vectors is in equilibrium.
 * Algorithm: Vector sum calculation. It reads 'n' vectors, sums their X, Y, and Z components,
 * and checks if all resulting sums are zero.
 * Time Complexity: O(N) where N is the number of vectors.
 * Space Complexity: O(1) as only a few variables are used to store the sums.
 */
#include <iostream>

using namespace std;

int main() {
  int many;
  cin >> many;
  int xt = 0, yt = 0, zt = 0;
  /**
   * @brief Reads 'many' sets of 3D vector components and accumulates their sums.
   * Pre-condition: 'many' specifies the total number of vectors to process.
   * 'xt', 'yt', 'zt' are initialized to zero to store the cumulative sums of x, y, and z components.
   * Invariant: At the start of each iteration, 'many' decreases, and 'xt', 'yt', 'zt' hold the sum
   * of components from previously processed vectors.
   */
  while (many--) {
    int x, y, z;
    cin >> x >> y >> z;
    xt += x;
    yt += y;
    zt += z;
  }
  /**
   * @brief Checks if the sum of all vector components is zero, indicating equilibrium.
   * Pre-condition: 'xt', 'yt', 'zt' contain the final sums of all x, y, and z components, respectively.
   * Post-condition: Prints "YES" if all sums are zero (equilibrium), otherwise "NO".
   */
  if (xt == yt && xt == zt && xt == 0) {
    cout << "YES" << endl;
    return 0;
  }
  cout << "NO" << endl;
  return 0;
}