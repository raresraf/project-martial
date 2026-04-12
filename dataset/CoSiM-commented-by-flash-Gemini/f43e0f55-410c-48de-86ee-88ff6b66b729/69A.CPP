/**
 * @file 69A.CPP
 * @brief Solution to the "Young Physicist" problem.
 *
 * This program determines if a system of 'n' forces acting on a body is in equilibrium.
 * Each force is represented by its components along the X, Y, and Z axes.
 * A body is in equilibrium if the sum of all forces in each direction (X, Y, Z) is zero.
 *
 * Algorithm: Vector Summation.
 * Time Complexity: O(N) for reading input and O(N) for summing components, resulting in O(N) total.
 * Space Complexity: O(N) to store the force components.
 */

#include <iostream>

using namespace std;

/**
 * @brief Main function to solve the Young Physicist problem.
 *
 * Functional Utility: Orchestrates input reading, force summation, and equilibrium check.
 */
int main() {
  int n, i, sumx = 0, sumy = 0, sumz = 0;
  int x[100] = {0}, y[100] = {0}, z[100] = {0};

  cin >> n;

  /**
   * Block Logic: Reads the components of 'n' force vectors from standard input.
   * Pre-condition: 'n' (number of force vectors) has been read.
   * Invariant: After each iteration, one force vector's components (x[i], y[i], z[i]) are stored.
   */
  for (i = 0; i < n; i++)
    cin >> x[i] >> y[i] >> z[i];

  /**
   * Block Logic: Computes the sum of all force components along each axis (X, Y, Z).
   * Pre-condition: All 'n' force vectors' components have been read.
   * Invariant: 'sumx', 'sumy', 'sumz' accumulate the total force along respective axes.
   */
  for (i = 0; i < n; i++) {
    sumx += x[i];
    sumy += y[i];
    sumz += z[i];
  }

  /**
   * Block Logic: Checks if the system of forces is in equilibrium.
   * Equilibrium Condition: The sum of forces in the X, Y, and Z directions must all be zero.
   * Post-condition: Prints "YES" if in equilibrium, "NO" otherwise.
   */
  if (sumx == 0 && sumy == 0 && sumz == 0)
    cout << "YES";
  else
    cout << "NO";

  return 0;
}
