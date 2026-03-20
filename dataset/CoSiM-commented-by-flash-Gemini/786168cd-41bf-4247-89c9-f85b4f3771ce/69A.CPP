/**
 * @file 69A.CPP
 * @brief Solution to Codeforces problem 69A - Young Physicist.
 *        This program determines if a body is in equilibrium given a set of force vectors.
 *        A body is in equilibrium if the sum of all force components (x, y, and z) is zero.
 * Algorithm:
 *   1. Read the number of force vectors, N.
 *   2. Initialize three accumulators (x, y, z) to zero for the sums of components.
 *   3. Loop N times, reading the x, y, and z components of each force vector.
 *   4. Add each component to its respective accumulator.
 *   5. After the loop, check if all three accumulators (x, y, and z) are simultaneously zero.
 *   6. Print "YES" if all sums are zero, indicating equilibrium; otherwise, print "NO".
 * Time Complexity: O(N), where N is the number of force vectors, as each vector is processed once.
 * Space Complexity: O(N) for storing the force components in the `arr` array. If processed directly
 *                   without storing all components, it could be O(1).
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  /**
   * @brief Main function to solve the "Young Physicist" problem.
   *
   * Reads the number of force vectors (N) and then processes N sets of
   * (x, y, z) force components. It accumulates the sum of components
   * along each axis and determines if the resultant force is zero in all dimensions.
   * @return int Returns 0 upon successful execution.
   */
  int N;
  cin >> N;

  int arr[N][3];

  int x = 0, y = 0, z = 0;

  // Block Logic: Iterates N times to read each force vector and accumulate its components.
  // Pre-condition: N specifies the total number of force vectors to process.
  // Invariant: x, y, and z maintain the running totals of force components.
  for (int i = 0; i < N; i++) {
    cin >> arr[i][0] >> arr[i][1] >> arr[i][2];
    x += arr[i][0];
    y += arr[i][1];
    z += arr[i][2];
  }

  // Block Logic: Checks if the system of forces is in equilibrium.
  // Pre-condition: x, y, and z hold the final resultant force components.
  // Post-condition: Prints "YES" if all components are zero (equilibrium), otherwise "NO".
  if (x == 0 and y == 0 and z == 0)
    cout << "YES";

  else
    cout << "NO";
}
