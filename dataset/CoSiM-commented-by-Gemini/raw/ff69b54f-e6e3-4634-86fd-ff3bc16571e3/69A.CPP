/**
 * @file 69A.CPP
 * @brief Solves the "Young Physicist" problem on Codeforces.
 *
 * @details This program determines if a body is in equilibrium under the influence
 * of multiple force vectors. A body is in equilibrium if the vector sum of all
 * forces is zero. This means the sum of the x, y, and z components of all
 * vectors must each independently be zero.
 *
 * Algorithm:
 * 1. Read the number of force vectors, n.
 * 2. Store all n vectors (each with 3 components) in a 2D array `a`.
 * 3. Iterate through each of the three dimensions (j = 0 to 2 for x, y, z).
 *    a. For the current dimension 'j', sum the components of all 'n' vectors.
 *    b. The sum is accumulated in a single variable 's' across all dimensions.
 *       This works because if the sum for a previous dimension was non-zero,
 *       the loop would have already terminated. If it was zero, it doesn't
 *       affect the sum for the current dimension.
 *    c. If the cumulative sum `s` is non-zero after summing a dimension's
 *       components, it means the forces are not balanced in that dimension.
 *       Set a flag `f` to 0 and break.
 * 4. If the flag `f` remains 1 (its initial state), it means the sum for all
 *    three dimensions was zero, so print "YES". Otherwise, print "NO".
 *
 * Time Complexity: O(N), for reading the N*3 matrix elements. The processing loop is also N*3.
 * Space Complexity: O(N) to store the vectors in the array `a`. (Note: fixed-size array `a[100][3]`).
 */
#include <iostream>
using namespace std;

int main() {
  int n, i, j, s = 0, a[100][3], f = 1;
  cin >> n;
  // Read all n * 3 vector components into the array.
  for (i = 0; i < n; i++)
    for (j = 0; j < 3; j++)
      cin >> a[i][j];

  /**
   * @brief Check for equilibrium by summing the force components along each axis.
   * Pre-condition: `a` contains all force vectors. `s` is 0. `f` (flag) is 1.
   * Invariant: The outer loop iterates through the three dimensions (x, y, z).
   * The variable `s` accumulates the sum for the current dimension. The flag `f`
   * is set to 0 if any dimensional sum is non-zero.
   */
  for (j = 0; j < 3; j++) {
    // Sum the components for the current dimension `j`.
    for (i = 0; i < n; i++)
      s += a[i][j];
    
    // If the sum for this dimension is non-zero, the body is not in equilibrium.
    // The check `if (s)` works because the sum for any previous dimension must have been
    // zero for the loop to continue.
    if (s) {
      f = 0;
      break;
    }
  }
  // If the flag is still 1, all dimensional sums were zero.
  if (f)
    cout << "YES";
  else
    cout << "NO";

  return 0;
}
