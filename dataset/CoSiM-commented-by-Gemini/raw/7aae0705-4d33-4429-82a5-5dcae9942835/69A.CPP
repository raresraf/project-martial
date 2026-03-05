/**
 * @file 69A.CPP
 * @brief Solution for Codeforces Problem 69A, "Young Physicist".
 * @details This program determines if a body is in equilibrium by checking if
 * the vector sum of all forces acting upon it is zero.
 *
 * Algorithm:
 * 1. Read the number of force vectors, `n`.
 * 2. Initialize three variables (`x1`, `y1`, `z1`) to store the sum of the
 *    x, y, and z components of all vectors, respectively.
 * 3. Loop `n` times to read each force vector.
 * 4. In each iteration, add the x, y, and z components of the current vector
 *    to the running sums `x1`, `y1`, and `z1`.
 * 5. After the loop, the triplet (`x1`, `y1`, `z1`) represents the resultant
 *    force vector.
 * 6. If `x1`, `y1`, and `z1` are all zero, the net force is zero, and the body
 *    is in equilibrium. The program prints "YES".
 * 7. Otherwise, the program prints "NO".
 *
 * Time Complexity: O(N), where N is the number of force vectors.
 * Space Complexity: O(1).
 */
#include <iostream>
using namespace std;
int n, x, y, z, x1 = 0, y1 = 0, z1 = 0;
int main() {
  cin >> n;
  /**
   * @brief This loop reads `n` force vectors and calculates the resultant
   * vector by summing up the components.
   */
  for (int i = 0; i < n; i++) {
    cin >> x >> y >> z;
    x1 += x;
    y1 += y;
    z1 += z;
  }
  /**
   * @brief Checks if the body is in equilibrium. This is true if and only if
   * the sum of the forces along each axis is zero.
   */
  if (x1 == 0 && y1 == 0 && z1 == 0) {
    cout << "YES";
  } else {
    cout << "NO";
  }
}
