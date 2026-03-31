/**
 * @file 69A.CPP
 * @brief Solves the "Young Physicist" problem using a single-pass modulo approach.
 *
 * This program checks if a body is in equilibrium by summing force vectors.
 * An object is in equilibrium if the sum of the x, y, and z components of all
 * forces are each zero. This implementation uses a clever single-loop approach.
 * It reads all `n * 3` force components sequentially and uses the modulo
 * operator (`i % 3`) to correctly assign each component to its corresponding
 * axis sum (x, y, or z), which are stored in a 3-element array.
 *
 * @author f4c177ce-4ae4-41e7-9622-16a479b5acab
 *
 * @algorithm Single-Pass Modulo-Based Summation
 * @complexity
 * Time: O(N), because the main loop runs `N * 3` times.
 * Space: O(1), as it only uses a fixed-size array of size 3.
 */
#include <cstdio>

// The number of axes in a 3D vector.
const int N = 3;
// `force` array stores the sum of components for each axis:
// force[0] for x-axis, force[1] for y-axis, force[2] for z-axis.
int force[N];

int main() {
  // `n` is the number of force vectors.
  int n;
  scanf("%d", &n);

  /**
   * @brief A single loop to read all 3*n components sequentially.
   * The modulo operator `i % 3` cleverly directs each component to the
   * correct axis accumulator in the `force` array.
   */
  for (int i = 0; i < n * N; ++i) {
    int m; // A temporary variable for the current component value.
    scanf("%d", &m);
    force[i % 3] += m;
  }

  bool ok = true; // Flag to track equilibrium status.

  /**
   * @brief Check if the sum of forces for each axis is zero.
   */
  for (int i = 0; i < N; ++i) {
    if (force[i] != 0) { // C-style check for non-zero.
      ok = false;
      break;
    }
  }

  // Use a ternary operator to print the result based on the `ok` flag.
  printf(ok ? "YES
" : "NO
");
  return 0;
}