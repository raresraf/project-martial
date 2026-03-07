/**
 * @file 69A.CPP
 * @brief Solution to the "Young Physicist" competitive programming problem (Codeforces 69A).
 * The problem states that an object is in equilibrium if the sum of all forces acting on it is zero.
 * The program reads the number of forces and then the x, y, and z components for each force.
 * It determines if the object is in equilibrium by checking if the sum of x-components,
 * sum of y-components, and sum of z-components are all independently equal to zero.
 * Algorithm: Matrix input and column-wise summation with conditional check.
 * Time Complexity: O(N), where N is the number of force vectors.
 * Space Complexity: O(N) for storing the force components.
 */
#include <iostream>
using namespace std;

int main() {
  // Declares 'n' for the number of force vectors.
  // Declares 'i' and 'j' as loop counters.
  int n, i, j;
  // Reads the number of force vectors.
  cin >> n;
  
  // Declares a 2D array 'arr' to store the x, y, z components of 'n' force vectors.
  // Each row represents a force vector, and columns represent x, y, z components.
  int arr[n][3];
  
  /**
   * Block Logic: Reads the x, y, z components for each of the 'n' force vectors.
   * Invariant: At the start of the outer loop, 'i' is the current force vector index.
   *            At the start of the inner loop, 'j' is the current component index (0=x, 1=y, 2=z).
   * Pre-condition: 'n' is a positive integer.
   * Post-condition: The 'arr' matrix is populated with force components.
   */
  for (i = 0; i < n; i++) {
    for (j = 0; j < 3; j++) {
      cin >> arr[i][j];
    }
  }
  
  // Declares a boolean variable 'ans' to store the equilibrium status, initialized to true.
  bool ans = true;
  
  /**
   * Block Logic: Iterates through each component (x, y, z) to calculate its sum across all force vectors.
   * Invariant: At the start of the outer loop, 'j' is the current component index (0=x, 1=y, 2=z).
   * Post-condition: 'ans' is false if any component sum is non-zero.
   */
  for (j = 0; j < 3; j++) {
    int sum = 0; // Initializes 'sum' for the current component to 0.
    /**
     * Block Logic: Sums the current component ('j') for all 'n' force vectors.
     * Invariant: At the start of the inner loop, 'i' is the current force vector index.
     * Pre-condition: 'arr' matrix is populated.
     */
    for (i = 0; i < n; i++) {
      sum += arr[i][j]; // Adds the component value to the sum.
    }
    // Conditional Logic: If the sum of the current component is not zero, the system is not in equilibrium.
    if (sum != 0) {
      ans = false; // Sets 'ans' to false.
    }
  }
  
  // Final Conditional Logic: Prints "YES" if the system is in equilibrium, "NO" otherwise.
  if (ans) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
  
  return 0; // Indicates successful program execution.
}
