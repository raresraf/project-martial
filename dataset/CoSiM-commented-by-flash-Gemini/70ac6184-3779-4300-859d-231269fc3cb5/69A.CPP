/**
 * @file 69A.CPP
 * @brief Solution to Codeforces Problem 69A: "Young Physicist".
 * Determines if a body is in equilibrium given 'n' force vectors. A body is in equilibrium
 * if the sum of all x-components, y-components, and z-components of the forces is zero.
 * Algorithm: Reads 'n' 3D force vectors, then sums their components along each axis.
 * The body is in equilibrium if all three sums are zero.
 * Time Complexity: O(N), where N is the number of force vectors.
 * Space Complexity: O(N) to store the force vectors.
 */
#include <iostream> // Required for input/output operations (cin, cout).
using namespace std;

/**
 * @brief Main function to solve the "Young Physicist" problem.
 * Reads force vectors, calculates the sum of their components, and determines
 * if the body is in equilibrium.
 */
int main() {
  int n; // Stores the number of force vectors.
  cin >> n; // Reads the total number of force vectors.

  /*
   * Commented-out section: This block shows an attempt at dynamic memory allocation
   * for a 2D array, which is not correctly implemented here and is bypassed by
   * the subsequent static array declaration. This part of the code is not active.
   *
   * int **arr=new int*[n];
   *
   * for(int i=0;i<n;i++)
   * {
   *         arr=new int[i][3];
   * }
   */

  // Declares a 2D array to store 'n' force vectors, each with 3 components (x, y, z).
  // This uses a Variable Length Array (VLA) which is a C99 feature and a GCC extension in C++.
  int arr[n][3];

  /**
   * Block Logic: Reads the components of 'n' force vectors from input.
   * Pre-condition: 'n' specifies the number of force vectors.
   * Invariant: After the outer loop, `arr` contains all `n` force vectors.
   *            After the inner loop, all 3 components of the current force vector are read.
   */
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 3; j++) {
      cin >> arr[i][j]; // Reads the x, y, or z component of the current force vector.
    }
  }

  // Initialize sums for x, y, and z components, and flags to track if sums are zero.
  int sum1 = 0, sum2 = 0, sum3 = 0;
  int flag1 = 0, flag2 = 0, flag3 = 0; // Using integers as boolean flags (0 for false, 1 for true).

  /**
   * Block Logic: Calculates the sum of all x-components of the force vectors.
   * Invariant: `sum1` accumulates the x-components from `arr[0][0]` to `arr[i][0]`.
   */
  for (int i = 0; i < n; i++) {
    sum1 += arr[i][0]; // Adds the x-component of the current force vector to sum1.
  }

  /**
   * Block Logic: Checks if the sum of x-components is zero.
   * If `sum1` is 0, sets `flag1` to indicate equilibrium along the x-axis.
   */
  if (sum1 == 0) {
    flag1 = 1; // Set flag1 to true.
  }

  /**
   * Block Logic: Conditionally calculates the sum of y-components.
   * This calculation only proceeds if `sum1` was zero (i.e., `flag1` is true),
   * representing an optimization to avoid unnecessary work if the body is already
   * not in equilibrium along the x-axis.
   */
  if (flag1 == 1) {
    for (int i = 0; i < n; i++) {
      sum2 += arr[i][1]; // Adds the y-component of the current force vector to sum2.
    }
  } else {
    // If sum1 is not zero, force sum2 to a non-zero value to ensure flag2 remains false.
    sum2 = -9; // Any non-zero value would work here.
  }

  /**
   * Block Logic: Checks if the sum of y-components is zero.
   * If `sum2` is 0, sets `flag2` to indicate equilibrium along the y-axis.
   */
  if (sum2 == 0) {
    flag2 = 1; // Set flag2 to true.
  }

  /**
   * Block Logic: Conditionally calculates the sum of z-components.
   * This calculation only proceeds if `sum2` was zero (i.e., `flag2` is true),
   * representing an optimization.
   */
  if (flag2 == 1) { // This condition is effectively 'if (sum1 == 0 && sum2 == 0)'.
    for (int i = 0; i < n; i++) {
      sum3 += arr[i][2]; // Adds the z-component of the current force vector to sum3.
    }
  }

  /**
   * Block Logic: Checks if the sum of z-components is zero.
   * If `sum3` is 0, sets `flag3` to indicate equilibrium along the z-axis.
   */
  if (sum3 == 0) {
    flag3 = 1; // Set flag3 to true.
  }

  /**
   * Block Logic: Determines the final equilibrium status.
   * If all three flags are true, the body is in equilibrium.
   */
  if ((flag1 && flag2) && flag3) { // Checks if all component sums are zero.
    cout << "YES" << endl;       // Body is in equilibrium.
  } else {
    cout << "NO" << endl;        // Body is not in equilibrium.
  }

  return 0;
}
