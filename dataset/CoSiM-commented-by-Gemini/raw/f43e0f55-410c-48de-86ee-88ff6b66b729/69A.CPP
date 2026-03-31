/**
 * @file 69A.CPP
 * @brief Solves the "Young Physicist" problem using a two-pass summation.
 *
 * This program checks if a physical body is in equilibrium. A body is in
 * equilibrium if the vector sum of all forces acting upon it is zero. This
 * requires the sum of the x, y, and z components of all forces to be
 * individually zero. This implementation first reads all `n` force vectors into
 * arrays and then, in a second pass, calculates the sum for each axis to
 * verify the equilibrium condition.
 *
 * @author f43e0f55-410c-48de-86ee-88ff6b66b729
 *
 * @algorithm Two-Pass Vector Component Summation
 * @complexity
 * Time: O(N), for iterating through the N vectors twice.
 * Space: O(N), to store the components of the N vectors in arrays.
 */
// 69A - Young Physicist

#include <iostream>

using namespace std;

int main() {
  // `n`: number of force vectors. `i`: loop counter.
  // `sumx`, `sumy`, `sumz`: accumulators for the component sums.
  int n, i, sumx = 0, sumy = 0, sumz = 0;
  // Arrays to store the components of up to 100 vectors.
  int x[100] = {0}, y[100] = {0}, z[100] = {0};

  cin >> n;

  /**
   * @brief First pass: Read all vector components into arrays.
   */
  for (i = 0; i < n; i++)
    cin >> x[i] >> y[i] >> z[i];

  /**
   * @brief Second pass: Sum the components for each axis.
   */
  for (i = 0; i < n; i++) {
    sumx += x[i];
    sumy += y[i];
    sumz += z[i];
  }

  /**
   * @brief Check for the equilibrium condition.
   * The condition is met if and only if the sum of forces for each axis is zero.
   */
  if (sumx == 0 && sumy == 0 && sumz == 0)
    cout << "YES";
  else
    cout << "NO";

  return 0;
}