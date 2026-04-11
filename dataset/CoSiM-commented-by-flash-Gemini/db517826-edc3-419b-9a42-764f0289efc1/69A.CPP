/**
 * @file 69A.CPP
 * @brief This program determines if a system of N vectors is in equilibrium.
 * It reads N 3-dimensional vectors (x, y, z components) and checks if the
 * sum of all x-components, all y-components, and all z-components are individually zero.
 *
 * Algorithm: Vector sum calculation. Iterate through components, accumulate sums, and check for zero.
 * Time Complexity: O(N) where N is the number of vectors.
 * Space Complexity: O(1) as only a fixed-size array stores the sum of components.
 */

#include <cstdio> // Required for scanf and printf

int main() {
  // Functional Utility: 'force' is an array to store the sum of x, y, and z components respectively.
  //                     Initialized to all zeros.
  int force[3] = {0};
  // Functional Utility: 'temp' temporarily stores each component read from input.
  int temp = 0;
  // Functional Utility: 'numVectors' stores the total number of 3D vectors.
  int numVectors;

  // Block Logic: Reads the number of vectors 'numVectors' from standard input.
  scanf("%d", &numVectors);

  /**
   * Block Logic: Reads all components of all vectors and accumulates their sums.
   * Invariant: After each iteration, 'temp' holds a component, and 'force' array
   *            is updated with the accumulated sum for that component type.
   *            The loop runs 3 * numVectors times because each vector has 3 components.
   */
  for (int k = 0; k < 3 * numVectors; k++) {
    // Block Logic: Reads a single component from standard input.
    scanf("%d", &temp);
    // Functional Utility: Adds the current component 'temp' to the corresponding
    //                     sum in the 'force' array. 'k % 3' maps to x, y, or z index.
    force[k % 3] += temp;
  }

  /**
   * Block Logic: Checks if the sum of all x, y, and z components are individually zero.
   * Pre-condition: 'force' array contains the total sums of all components.
   * Invariant: If all sums are zero, the system is in equilibrium ("YES"), otherwise ("NO").
   * Inline: The logical OR operator checks if any of the sums are non-zero. If any are non-zero,
   *         the expression (force[0] || force[1] || force[2]) evaluates to true (non-zero),
   *         resulting in "NO". Otherwise, it's "YES".
   */
  (force[0] || force[1] || force[2]) ? printf("NO") : printf("YES");
  return 0;
}
