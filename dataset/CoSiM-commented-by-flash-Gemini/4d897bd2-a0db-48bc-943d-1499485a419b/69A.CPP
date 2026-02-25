/**
 * @file 69A.CPP
 * @brief Solves the "Young Physicist" problem from Codeforces (Problem 69A).
 *        This program determines if a system of N forces acting on an object
 *        is in equilibrium. Equilibrium is achieved if the sum of all force
 *        vectors is the zero vector.
 * Algorithm: Vector Summation
 * Time Complexity: O(N), where N is the number of force vectors.
 * Space Complexity: O(N) to store the force components.
 */

#include <iostream>
#include <string>
#include <vector>

using namespace std;

/**
 * @brief Main function to read force vectors, calculate their sum,
 *        and determine if the object is in equilibrium.
 * @return 0 if the program executes successfully.
 */
int main(void) {
  // Functional Utility: Optimizes C++ standard streams for competitive programming.
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  // Declare an integer to store the number of force vectors.
  int numberOfVectors;
  // Read the number of force vectors from standard input.
  cin >> numberOfVectors;

  // Declare a 2D array to store the x, y, and z components of each force vector.
  // The size is dynamically allocated based on the input 'numberOfVectors'.
  int arr[numberOfVectors][3];

  /**
   * Block Logic: Reads the components of each force vector.
   * Invariant: After each iteration, one force vector's components are stored.
   */
  for (int i = 0; i < numberOfVectors; i++)
    // Read the x, y, and z components for the current vector.
    cin >> arr[i][0] >> arr[i][1] >> arr[i][2];

  // Initialize accumulators for the sum of x, y, and z components.
  int p = 0, q = 0, r = 0;
  /**
   * Block Logic: Calculates the sum of all x, y, and z components across all vectors.
   * Invariant: After each iteration, 'p', 'q', and 'r' hold the cumulative sum
   *            of the respective components up to the current vector.
   */
  for (int i = 0; i < numberOfVectors; i++) {
    // Accumulate the x-components.
    p += arr[i][0];
    // Accumulate the y-components.
    q += arr[i][1];
    // Accumulate the z-components.
    r += arr[i][2];
  }

  /**
   * Block Logic: Checks if the resultant force vector is the zero vector.
   * Pre-condition: 'p', 'q', and 'r' contain the sums of all vector components.
   * Post-condition: Prints "YES" if the object is in equilibrium, "NO" otherwise.
   */
  if (p == 0 && q == 0 && r == 0)
    cout << "YES" << endl; // Output "YES" if the sum of forces is zero (equilibrium).
  else
    cout << "NO" << endl;  // Output "NO" otherwise.

  return 0; // Indicate successful execution.
}