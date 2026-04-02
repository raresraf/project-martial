/**
 * @file 69A.CPP
 * @brief Solution to the "Young Physicist" problem (Codeforces 69A).
 *
 * Problem Statement:
 * A body is in equilibrium if the sum of all forces acting on it is the zero vector.
 * Given 'n' force vectors, each specified by its x, y, and z components, the task
 * is to determine if the body is in equilibrium.
 *
 * Algorithm:
 * The principle of equilibrium states that the net force must be zero. This is true
 * if and only if the sum of the components along each axis (x, y, and z) is zero.
 * This program implements this check directly. It reads 'n' vectors, storing them
 * in a 2D array. As it reads each vector, it adds its components to three running
 * totals: `x`, `y`, and `z`. After processing all vectors, it checks if `x`, `y`,
 * and `z` are all equal to zero. If they are, it confirms equilibrium ("YES");
 * otherwise, it reports that the body is not in equilibrium ("NO").
 *
 * Note on Implementation:
 * The code uses a 2D array `vec[100][3]` to store all vector components. This is
 * not strictly necessary, as the sums could be computed without storing the
 * entire array, but it doesn't affect the correctness of the result.
 *
 * Time Complexity: O(n), as it requires a single pass to read the 'n' vectors.
 * Space Complexity: O(1), as the array size is fixed and does not depend on 'n'.
 */
#include <iostream>
using namespace std;

int main() {
  int n;
  // Initialize sums for x, y, and z components to zero.
  int x = 0, y = 0, z = 0;
  // A 2D array to store the force vectors.
  int vec[100][3];
  cin >> n;
  
  /**
   * Block Logic: Iterate through 'n' vectors, reading their components
   * and adding them to the corresponding total sums.
   */
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 3; j++)
      cin >> vec[i][j];
    x += vec[i][0];
    y += vec[i][1];
    z += vec[i][2];
  }

  /**
   * Block Logic: Check if the sum of forces along each axis is zero.
   * This is the condition for static equilibrium.
   */
  if (x == 0 && y == 0 && z == 0)
    cout << "YES";
  else
    cout << "NO";
  return 0;
}
