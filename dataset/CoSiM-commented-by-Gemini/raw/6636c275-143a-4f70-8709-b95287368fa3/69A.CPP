/**
 * @file 69A.CPP
 * @brief Solves the "Young Physicist" problem on Codeforces (Problem 69A).
 * @details This program determines if a body is in equilibrium under the influence
 * of multiple force vectors. According to Newton's First Law, a body is in
 * equilibrium if and only if the net force acting upon it is zero. This means
 * the vector sum of all forces must be the zero vector (0, 0, 0).
 *
 * Algorithm: Component-wise Vector Summation.
 * The program reads all the force vectors and calculates the sum of their
 * x, y, and z components separately. If all three sums are zero, the body
 * is in equilibrium.
 *
 * Time Complexity: O(N), where N is the number of force vectors, due to a single
 * pass to read and sum the vectors.
 * Space Complexity: O(1), as only a few variables are used to store the running sums.
 */
#include <stdio.h>

int main(void) {
  // x, y, z: components of a single force vector read from input.
  // sum_x, sum_y, sum_z: running sums of the components of the net force vector.
  // n: the total number of force vectors to be read.
  int x, y, z;
  int sum_x = 0;
  int sum_y = 0;
  int sum_z = 0;
  int n = 0;

  // Read the number of vectors.
  scanf("%d", &n);

  /**
   * @brief Block Logic: Reads each force vector and accumulates its components.
   * Invariant: After each iteration, sum_x, sum_y, and sum_z hold the net
   * force components for all vectors processed so far.
   */
  for (int i = 0; i < n; i++) {
    scanf("%d %d %d", &x, &y, &z);

    // Add the components of the current vector to the total sums.
    sum_x += x;
    sum_y += y;
    sum_z += z;
  }

  /**
   * @brief Block Logic: Checks the equilibrium condition.
   * The body is in equilibrium if the net force vector is (0, 0, 0),
   * which requires each component sum to be zero.
   */
  if (sum_x == 0 && sum_y == 0 && sum_z == 0) {
    printf("YES");
  } else {
    printf("NO");
  }
  return 0;
}
