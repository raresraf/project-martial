/**
 * @file 69A.CPP
 * @brief Solution to Codeforces Problem 69A: "Young Physicist".
 *
 * This problem states that a body is in equilibrium if the sum of all forces
 * acting on it is the zero vector. Given `N` force vectors, each with `x, y, z`
 * components, the task is to determine if the body is in equilibrium.
 *
 * Algorithm: Initialize three accumulators for the x, y, and z components of forces to zero.
 * Read each force vector, adding its components to the respective accumulators.
 * After processing all forces, check if all three accumulated sums are zero.
 *
 * Time Complexity: O(N), where N is the number of force vectors. Each force vector
 *                  is processed in constant time.
 * Space Complexity: O(1) as only a few variables are used to store sums.
 */
#include <algorithm> // Included but not explicitly used for the core logic in this solution.
#include <cmath>     // Included but not explicitly used for the core logic in this solution.
#include <cstdio>    // Required for `scanf`, `printf`.
#include <cstring>   // Included but not explicitly used for the core logic in this solution.
#include <iostream>  // Included but not explicitly used for the core logic in this solution.
#include <map>       // Included but not explicitly used for the core logic in this solution.
#include <queue>     // Included but not explicitly used for the core logic in this solution.
#include <stack>     // Included but not explicitly used for the core logic in this solution.
#include <string>    // Included but not explicitly used for the core logic in this solution.
#include <utility>   // Included but not explicitly used for the core logic in this solution.
#include <vector>    // Included but not explicitly used for the core logic in this solution.

/** @def fi
 * @brief Alias for `first`, commonly used with `std::pair`.
 */
#define fi first
/** @def se
 * @brief Alias for `second`, commonly used with `std::pair`.
 */
#define se second
/** @def pb
 * @brief Alias for `push_back`, commonly used with `std::vector`.
 */
#define pb push_back
/** @def mp
 * @brief Alias for `make_pair`, commonly used with `std::pair`.
 */
#define mp make_pair
/** @def pi
 * @brief Defines the value of PI (2 * arccos(0.0) approximates PI).
 */
#define pi 2 * acos(0.0)
/** @def eps
 * @brief A small epsilon value, typically used for floating-point comparisons.
 */
#define eps 1e-9
/** @def PII
 * @brief Type alias for `std::pair<int, int>`.
 */
#define PII pair<int, int>
/** @def PDD
 * @brief Type alias for `std::pair<double, double>`.
 */
#define PDD pair<double, double>
/** @def LL
 * @brief Type alias for `long long`.
 */
#define LL long long

using namespace std;

int N;  /**< @brief Stores the number of force vectors. */
int x, y, z; /**< @brief Temporary variables to store x, y, z components of a force vector. */
int fx, fy, fz; /**< @brief Accumulators for the sum of x, y, z components across all forces. */

/**
 * @brief Main function to solve the "Young Physicist" problem.
 * Reads the number of force vectors and their components, calculates the
 * resultant force vector, and determines if the body is in equilibrium.
 */
int main() {
  scanf("%d", &N); // Reads the total number of force vectors.
  fx = fy = fz = 0; // Initialize accumulators for force components to zero.

  /**
   * Block Logic: Iterates through each force vector, summing its components.
   * Pre-condition: `N` specifies the number of force vectors. `fx`, `fy`, `fz` are 0.
   * Invariant: After each iteration, `fx`, `fy`, `fz` hold the accumulated sums of
   *            x, y, z components for the forces processed so far.
   */
  while (N--) { // Loop `N` times.
    scanf("%d %d %d", &x, &y, &z); // Reads the x, y, and z components of the current force vector.
    fx += x; // Accumulate x-components.
    fy += y; // Accumulate y-components.
    fz += z; // Accumulate z-components.
  }

  /**
   * Block Logic: Determines if the body is in equilibrium.
   * The body is in equilibrium if and only if the sum of all x, y, and z
   * components (`fx`, `fy`, `fz`) are each equal to zero.
   * The `!` operator negates the boolean value (0 becomes 1, non-zero becomes 0).
   */
  if (!fx && !fy && !fz) { // Checks if fx == 0 AND fy == 0 AND fz == 0.
    printf("YES\n");      // Body is in equilibrium.
  } else {
    printf("NO\n");       // Body is not in equilibrium.
  }
  return 0; // Indicates successful execution.
}
