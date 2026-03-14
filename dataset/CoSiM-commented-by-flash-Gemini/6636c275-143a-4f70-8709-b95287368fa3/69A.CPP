/**
 * @6636c275-143a-4f70-8709-b95287368fa3/69A.CPP
 * @brief This program solves Codeforces problem 69A, "Young Physicist".
 * It determines if an object is in equilibrium given a set of forces acting upon it.
 * Equilibrium is achieved if the sum of all force components along the X, Y, and Z axes
 * are all simultaneously equal to zero.
 *
 * Algorithm: Vector sum. Reads 'n' 3D force vectors, accumulates their components,
 * and then checks if all resultant components are zero.
 * Time Complexity: O(N) where N is the number of force vectors.
 * Space Complexity: O(1) for storing cumulative sums.
 */
#include <stdio.h> // For standard input/output functions (scanf, printf).

/**
 * @brief Main function to determine if an object is in equilibrium.
 * Reads 'n' force vectors, sums their components, and checks if all sums are zero.
 * @return 0 on successful execution.
 */
int main(void) {
  int x, y, z;       // Declares variables to store the x, y, and z components of each force vector.
  int sum_x = 0;     // Accumulates the sum of all x-components.
  int sum_y = 0;     // Accumulates the sum of all y-components.
  int sum_z = 0;     // Accumulates the sum of all z-components.
  int n = 0;         // Stores the number of force vectors.

  scanf("%d", &n); // Reads the number of force vectors from standard input.

  /**
   * Block Logic: Reads 'n' force vectors and accumulates their components.
   * Invariant: `sum_x`, `sum_y`, and `sum_z` hold the cumulative sum of force
   * components along each axis after processing each vector.
   */
  for (int i = 0; i < n; i++) { // Loops 'n' times, once for each force vector.
    // Reads the x, y, and z components of the current force vector.
    scanf("%d %d %d", &x, &y, &z); 

    sum_x += x; // Adds the current x-component to the total x-sum.
    sum_y += y; // Adds the current y-component to the total y-sum.
    sum_z += z; // Adds the current z-component to the total z-sum.
  }

  /**
   * Block Logic: Checks if the object is in equilibrium.
   * An object is in equilibrium if all three cumulative sums (x, y, z) are zero.
   * Pre-condition: `sum_x`, `sum_y`, `sum_z` contain the final sums of all force components.
   * Invariant: Prints "YES" if equilibrium, "NO" otherwise.
   */
  if (sum_x == 0 && sum_y == 0 && sum_z == 0) { // Checks if all component sums are zero.
    printf("YES"); // Prints "YES" if in equilibrium.
  } else {
    printf("NO"); // Prints "NO" otherwise.
  }
  return 0; // Indicates successful program termination.
}
