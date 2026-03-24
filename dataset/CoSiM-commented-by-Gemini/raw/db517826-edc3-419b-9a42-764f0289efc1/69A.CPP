/**
 * @file 69A.CPP
 * @brief Solution for Codeforces Problem 69A - "Young Physicist".
 *
 * @details
 * This program determines if a body is in equilibrium under the influence of
 * multiple force vectors. A body is in equilibrium if the vector sum of all
 * forces acting on it is the zero vector. This means the sum of the x-components,
 * the sum of the y-components, and the sum of the z-components must each be zero.
 *
 * Algorithm:
 * 1. An array `force[3]` is used to maintain the running sum of the x, y, and z components.
 * 2. The program reads the number of force vectors, `n`.
 * 3. It then iterates `3 * n` times, reading each component of each vector.
 * 4. Using the modulo operator (`k % 3`), each component is added to the correct
 *    total (x-sum, y-sum, or z-sum) in the `force` array.
 * 5. Finally, it checks if all three sums in the `force` array are zero. If they are,
 *    the body is in equilibrium ("YES"); otherwise, it is not ("NO").
 *
 * Time Complexity: O(N), where N is the number of vectors, as the code iterates 3*N times.
 * Space Complexity: O(1), as storage is constant regardless of the number of vectors.
 */
#include <cstdio>

int main() {
  // `force` array stores the sum of x, y, and z components at indices 0, 1, and 2 respectively.
  int force[3] = {0};
  // `temp` holds the value of each force component as it's read.
  int temp = 0;
  // `numVectors` is the total number of force vectors.
  int numVectors;
  scanf("%d", &numVectors);
  
  /**
   * Block Logic: Reads all 3*N vector components and calculates the sum for each axis.
   * This loop cleverly avoids a nested structure by iterating through all components at once.
   * The `k % 3` expression correctly maps each input value to its corresponding axis (x, y, or z).
   */
  for (int k = 0; k < 3 * numVectors; k++) {
    scanf("%d", &temp);
    // Invariant: `force[k % 3]` accumulates the sum for the current axis.
    force[k % 3] += temp;
  }
  
  // Functional Utility: Checks the equilibrium condition.
  // If the sum of forces along any axis is non-zero, the net force is non-zero,
  // and the body is not in equilibrium.
  (force[0] || force[1] || force[2]) ? printf("NO") : printf("YES");
  return 0;
}
