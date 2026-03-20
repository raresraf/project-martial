/**
 * @file 69A.CPP
 * @brief A concise, C-style solution for Codeforces problem 69A, "Young Physicist".
 * @details This program determines if a body is in equilibrium by checking if the
 * vector sum of all applied forces is zero. It uses a clever loop structure
 * to process all force components.
 *
 * @section Algorithm
 * The core principle is that for a body to be in equilibrium, the sum of the
 * components of all force vectors along each axis (x, y, z) must be zero.
 *
 * This implementation uses a single loop to read all `3*n` components:
 * 1. An integer array `force[3]` is initialized to zero to store the sum of
 *    forces for the x, y, and z axes.
 * 2. The number of force vectors, `numVectors`, is read.
 * 3. A `for` loop iterates `3 * numVectors` times. In each iteration, it reads
 *    one integer component.
 * 4. The modulo operator (`k % 3`) is used as an index to add the component to
 *    the correct accumulator in the `force` array (`force[0]` for x, `force[1]`
 *    for y, `force[2]` for z).
 * 5. After the loop, a ternary operator checks if any of the component sums are
 *    non-zero. If `force[0] || force[1] || force[2]` is true, it means there is
 *    a net force, and it prints "NO". Otherwise, it prints "YES".
 *
 * @subsection Complexity
 * - Time Complexity: O(n), as the loop iterates `3*n` times, which is linear in `n`.
 * - Space Complexity: O(1), as it uses a fixed-size array and a few variables.
 */
#include <cstdio>

int main() {
  // `force` array stores the sum of components for the x, y, and z axes.
  int force[3] = {0};
  // `temp` holds the value of each component as it's read.
  int temp = 0;
  // `numVectors` is the number of force vectors.
  int numVectors;
  scanf("%d", &numVectors);

  /**
   * @brief This single loop reads all 3*n vector components sequentially.
   * The `k % 3` expression cleverly routes each component value to the correct
   * accumulator in the `force` array (0 for x, 1 for y, 2 for z).
   */
  for (int k = 0; k < 3 * numVectors; k++) {
    scanf("%d", &temp);
    force[k % 3] += temp;
  }

  /**
   * @brief This ternary expression provides a concise way to check for equilibrium.
   * The condition `(force[0] || force[1] || force[2])` evaluates to true if
   * any of the net force components is non-zero, meaning the body is not in
   * equilibrium.
   */
  (force[0] || force[1] || force[2]) ? printf("NO") : printf("YES");
  return 0;
}
