/**
 * @file 69A.CPP
 * @brief Solution for Codeforces Problem 69A: Young Physicist.
 * @details This program determines if a system of 'n' forces acting on an object is in equilibrium.
 * It reads 'n' sets of three-dimensional force vectors (x, y, z components) and sums their components
 * along each axis. The system is in equilibrium if and only if the sum of force components along
 * the X-axis, Y-axis, and Z-axis are all zero.
 * @algorithm Summation of 3D vectors.
 * Time Complexity: $O(N)$ where N is the number of force vectors.
 * Space Complexity: $O(1)$ as only a fixed-size array is used to store force sums.
 */
#include <cstdio>

/**
 * @file 69A.CPP
 * @brief Solution for Codeforces Problem 69A: Young Physicist.
 * @details This program determines if a system of 'n' forces acting on an object is in equilibrium.
 * It reads 'n' sets of three-dimensional force vectors (x, y, z components) and sums their components
 * along each axis. The system is in equilibrium if and only if the sum of force components along
 * the X-axis, Y-axis, and Z-axis are all zero.
 * @algorithm Summation of 3D vectors.
 * Time Complexity: $O(N)$ where N is the number of force vectors.
 * Space Complexity: $O(1)$ as only a fixed-size array is used to store force sums.
 */
#include <cstdio>

/**
 * @brief Defines the dimensionality of the force vectors (3 for x, y, z components).
 */
const int N = 3;

/**
 * @brief Array to store the cumulative sum of force components along each axis.
 * @details `force[0]` for X-axis, `force[1]` for Y-axis, `force[2]` for Z-axis.
 * Initialized to all zeros implicitly.
 */
int force[N];

int main() {
  /**
   * @brief Reads the total number of force vectors to be processed.
   * @param n The number of force vectors.
   */
  int n;
  scanf("%d", &n);

  /**
   * @brief Reads 'n' 3D force vectors and accumulates their components.
   * @block_logic Iterates `n * N` times to read `n` vectors, each with `N` components.
   * @invariant At the end of each iteration, `force[i % 3]` holds the sum of the
   *            (i % 3)-th component of all processed force vectors so far.
   */
  for (int i = 0; i < n * N; ++i) {
    int m;
    scanf("%d", &m);
    force[i % 3] += m; // Accumulates force component for X, Y, or Z axis.
  }

  /**
   * @brief Determines if the system of forces is in equilibrium.
   * @details A system is in equilibrium if the sum of forces along each axis is zero.
   * @block_logic Checks if any component sum in the `force` array is non-zero.
   * @invariant `ok` remains true if all component sums are zero; becomes false otherwise.
   */
  bool ok = true;

  for (int i = 0; i < N; ++i) {
    if (force[i]) { // Check if the sum of forces along this axis is non-zero.
      ok = false;
      break; // If any axis sum is non-zero, the system is not in equilibrium.
    }
  }

  /**
   * @brief Prints the result based on whether the system is in equilibrium.
   * @details Prints "YES" if `ok` is true (system in equilibrium), otherwise "NO".
   */
  printf(ok ? "YES\n" : "NO\n");
  return 0;
}
