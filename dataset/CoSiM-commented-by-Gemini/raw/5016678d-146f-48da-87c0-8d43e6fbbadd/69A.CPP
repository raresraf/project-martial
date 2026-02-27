/**
 * @file 69A.CPP
 * @brief Solves the "Young Physicist" problem from Codeforces.
 * @details The problem checks if a body is in equilibrium under the influence
 * of several force vectors. For a body to be in equilibrium, the vector
 * sum of all forces acting on it must be the zero vector.
 *
 * @algorithm
 * 1. Read the number of force vectors, n.
 * 2. Initialize three variables (sumx, sumy, sumz) to zero to store the
 *    sum of the components of all force vectors along the X, Y, and Z axes.
 * 3. Loop n times:
 *    a. Read the x, y, and z components of a force vector.
 *    b. Add these components to their respective sums.
 * 4. After the loop, check if sumx, sumy, and sumz are all equal to 0.
 * 5. If all sums are 0, the body is in equilibrium, so print "YES".
 *    Otherwise, print "NO".
 *
 * @complexity
 * Time Complexity: O(n), where n is the number of force vectors, because we
 * iterate through the input vectors once.
 * Space Complexity: O(1), as we only use a few variables to store the sums,
 * regardless of the number of vectors.
 */
#include <iostream>
using namespace std;

main() {
  int n; // Number of force vectors.
  // Initialize sums of vector components for each axis to zero.
  int sumx = 0, sumy = 0, sumz = 0;
  cin >> n;

  /**
   * @block
   * @brief Loop through all force vectors, accumulating their components.
   * @invariant At the end of each iteration, sumx, sumy, and sumz hold the
   * component-wise sum of the vectors processed so far.
   */
  while (n > 0) {
    int x, y, z; // Components of the current force vector.
    cin >> x >> y >> z;
    sumx += x;
    sumy += y;
    sumz += z;
    n -= 1;
  }

  /**
   * @brief Check for equilibrium condition.
   * @details If the net force vector is (0, 0, 0), the body is in
   * equilibrium. This is true if and only if the sum of components along
   * each axis is zero.
   */
  if (sumx == 0 && sumy == 0 && sumz == 0)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}
