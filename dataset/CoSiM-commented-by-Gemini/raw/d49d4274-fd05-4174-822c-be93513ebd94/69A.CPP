/**
 * @file 69A.CPP
 * @brief Solution to the "Young Physicist" problem from Codeforces.
 * @details This program determines if a body is in equilibrium by summing up force vectors.
 * It reads an integer 'n', the number of forces acting on a body. Then, for each force,
 * it reads its x, y, and z components. The condition for equilibrium is that the vector
 * sum of all forces is zero. This is checked by summing the components of all vectors
 * along each of the three axes (x, y, z) independently. If the sum of components
 * along each axis is zero, the body is in equilibrium.
 *
 * @algorithm Vector Summation
 * @complexity
 * Time: O(N), where N is the number of force vectors, as it iterates through each vector once.
 * Space: O(1), as it uses a fixed number of variables to store the sums of the components.
 */
// By archit-dev, contest: Codeforces Beta Round #63 (Div. 2), problem: (A)
// Young Physicist, Accepted, #, Copy

#include <iostream>

using namespace std;

int main() {
  // n: number of force vectors.
  // x, y, z: components of a single force vector.
  // xsum, ysum, zsum: sum of components of all force vectors along each axis.
  int n, x, y, z, xsum(0), ysum(0), zsum(0);

  // Read the number of force vectors.
  cin >> n;

  /**
   * @brief Loop to read each force vector and update the component sums.
   * Pre-condition: 'n' is the number of vectors to be read.
   * Invariant: At the end of each iteration, xsum, ysum, and zsum hold the
   * sum of components for all vectors read so far.
   */
  while (n--) {
    cin >> x >> y >> z;
    xsum += x;
    ysum += y;
    zsum += z;
  }

  /**
   * @brief Check for equilibrium condition.
   * The body is in equilibrium if and only if the net force is zero.
   * This is true if the sum of vector components along each axis is zero.
   */
  if (xsum == 0 && ysum == 0 && zsum == 0) {
    cout << "YES
";
  } else {
    cout << "NO
";
  }

  return 0; // Indicate successful execution.
}