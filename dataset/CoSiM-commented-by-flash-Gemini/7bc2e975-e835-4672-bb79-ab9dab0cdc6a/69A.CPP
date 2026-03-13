/**
 * @file 69A.CPP
 * @brief Solves the "Young Physicist" problem from Codeforces (69A).
 * @details This program determines if a body is in equilibrium given 'n' force
 * vectors acting upon it.
 *
 * @section Algorithm
 * The core principle is that a body is in equilibrium if and only if the vector
 * sum of all forces acting on it is the zero vector. In a 3D coordinate system,
 * this means the sum of the x-components, the sum of the y-components, and the
 * sum of the z-components must all independently be zero.
 *
 * The algorithm proceeds as follows:
 * 1. Read the number of force vectors, 'n'.
 * 2. Read all 'n' vectors, storing their x, y, and z components.
 * 3. Calculate the sum of all x-components, the sum of all y-components, and
 *    the sum of all z-components.
 * 4. If all three sums are equal to zero, the body is in equilibrium, and the
 *    program outputs "YES". Otherwise, it outputs "NO".
 *
 * @section Complexity
 * - Time Complexity: O(n), because the program iterates through the 'n' vectors twice
 *   (once for input, once for summation).
 * - Space Complexity: O(n), due to the storage of vector components in three arrays
 *   of size 'n'. A more space-efficient O(1) approach could sum the components
 *   as they are read, avoiding storage.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {

  // 'n' is the number of force vectors.
  int n;
  cin >> n;

  /**
   * @var sum1, sum2, sum3 Store the sum of components along the x, y, and z axes, respectively.
   * @var x, y, z Arrays to store the components of the 'n' force vectors.
   * Note: Declaring arrays as `[n + 10]` is a common competitive programming habit
   * to add a safety buffer and prevent potential off-by-one errors, though not
   * strictly necessary for this problem's logic.
   */
  int sum1 = 0, sum2 = 0, sum3 = 0, x[n + 10], y[n + 10], z[n + 10];

  /**
   * @brief Block Logic: Reads all 'n' force vectors from standard input and stores
   * their components in the x, y, and z arrays.
   */
  for (int i = 0; i < n; i++)
    cin >> x[i] >> y[i] >> z[i];

  /**
   * @brief Block Logic: Iterates through the stored vectors to calculate the
   * component-wise sum of all forces.
   */
  for (int i = 0; i < n; i++) {
    sum1 += x[i];
    sum2 += y[i];
    sum3 += z[i];
  }

  /**
   * @brief Logic: Checks for equilibrium. The condition is met if the net force
   * along each of the three axes is zero.
   */
  if (sum1 == 0 && sum2 == 0 && sum3 == 0)
    cout << "YES" << endl; // The body is in equilibrium.
  else
    cout << "NO" << endl;  // The body is not in equilibrium.

  return 0;
}