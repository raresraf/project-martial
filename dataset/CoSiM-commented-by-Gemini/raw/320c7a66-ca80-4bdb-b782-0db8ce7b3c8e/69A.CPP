/**
 * @file 69A.CPP
 * @brief Solution to a competitive programming problem checking for equilibrium of forces.
 * This program reads a specified number of 3D force vectors, stores their components
 * in a 2D array, and calculates the sum of each component. It then determines if
 * all component sums are zero, indicating a state of equilibrium.
 *
 * Algorithm: Array-based summation of vector components.
 * Time Complexity: O(N) where N is the number of force vectors.
 * Space Complexity: O(N) for storing vector components in a 2D array.
 */
#include <iostream>

using namespace std;

/**
 * @brief Main function of the program.
 * Reads the number of force vectors `num`, then processes `num` vectors by storing
 * their x, y, and z components into a 2D array. It then calculates the sum of
 * all x, y, and z components separately to check for equilibrium.
 * @return `0` upon successful execution.
 */
int main() {
  int x = 0, y = 0, z = 0;
  int num;
  cin >> num;
  // Functional Utility: Declares a Variable-Length Array (VLA) to store `num` force vectors,
  // each with 3 components (x, y, z).
  int coordinates[num][3];
  /**
   * @brief Loop to read `num` force vectors and accumulate their components.
   * Invariant: The outer loop iterates `num` times (once for each vector), and the
   * inner loop reads the 3 components (x, y, z) of the current vector.
   * After reading, the components are added to their respective running sums (`x`, `y`, `z`).
   */
  for (int i = 0; i < num; i++) {
    for (int j = 0; j < 3; j++) {
      cin >> coordinates[i][j];
    }
    x = x + coordinates[i][0];
    y = y + coordinates[i][1];
    z = z + coordinates[i][2];
  }
  /**
   * @brief Checks for the equilibrium condition.
   * An object is in equilibrium if the sum of all x, y, and z components (`x`, `y`, `z` respectively) are all zero,
   * the object is in equilibrium ("YES"). Otherwise, it is not ("NO").
   */
  if (x == 0 && y == 0 && z == 0) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  return 0;
}

