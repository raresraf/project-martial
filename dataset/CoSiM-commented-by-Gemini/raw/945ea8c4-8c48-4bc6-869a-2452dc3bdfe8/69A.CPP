/**
 * @file 69A.CPP
 * @brief Determines if a system of 3D vectors sums to zero (is in equilibrium).
 *
 * This program solves the "Young Physicist" problem from Codeforces. It reads
 * `n` 3D vectors and calculates the sum of their x, y, and z components.
 * If the resulting net vector is (0, 0, 0), it indicates the system is in
 * equilibrium.
 *
 * @algorithm Vector Summation
 * @complexity
 * - Time: O(N), where N is the number of vectors, as it processes each vector once.
 * - Space: O(1), ignoring the fixed-size input array. The actual space needed is
 *   constant.
 */
#include <bits/stdc++.h>
#include <iostream>

using namespace std;

int main() {
  // An array to temporarily store the components of each vector.
  // Note: Declared with size 100, but only the first 3 elements are used.
  int x[100];
  // n: the number of vectors to process.
  int n;
  cin >> n;

  // result, result1, result2: store the running sum of x, y, and z components.
  int result = 0, result1 = 0, result2 = 0;

  /**
   * @brief Block Logic: Reads `n` vectors and accumulates their components.
   * Invariant: After each iteration of the outer loop, the `result` variables
   *            hold the vector sum of all vectors processed so far.
   */
  while (n--) {
    // Read the 3 components of a single vector.
    for (int i = 0; i < 3; i++) {
      cin >> x[i];
    }
    // Add the components to their respective totals.
    result = result + x[0];
    result1 = result1 + x[1];
    result2 = result2 + x[2];
  }

  /**
   * @brief Block Logic: Checks if the system is in equilibrium.
   * Pre-condition: The sums of all x, y, and z components have been calculated.
   */
  if (result == 0 && result1 == 0 && result2 == 0) {
    cout << "YES";
  } else {
    cout << "NO";
  }

  return 0;
}