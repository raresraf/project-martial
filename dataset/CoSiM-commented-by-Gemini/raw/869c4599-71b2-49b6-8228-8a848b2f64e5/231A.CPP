/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem from competitive programming.
 *
 * @details
 * This program counts how many problems a team of three will solve. A problem is
 * solved if at least two members are confident. This implementation uses a 2D
 * variable-length array to store the inputs, but processes the logic for each
 * problem immediately after its inputs are read.
 *
 * Algorithm:
 * 1. Read the number of problems, n.
 * 2. Initialize a final counter, `output`, to 0.
 * 3. Use a variable-length array (VLA), `matx[n][3]`, to store inputs. Note that
 *    VLAs are a non-standard C++ extension.
 * 4. Loop `n` times for each problem:
 *    a. Initialize a local counter for confident members for the current problem.
 *    b. In an inner loop, read the three confidence levels, store them in the
 *       matrix, and increment the local counter if the value is 1.
 *    c. After reading the three values, if the local counter is 2 or more,
 *       increment the final `output` counter.
 * 5. Print the final `output`.
 *
 * Time Complexity: O(N), for the single pass through N*3 inputs.
 * Space Complexity: O(N), due to the `matx[n][3]` array.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, output = 0; // n: number of problems, output: final count of solved problems.
  cin >> n;
  // A variable-length array (VLA) is used here. This is a non-standard C++ extension.
  // The matrix is used to store inputs, though it's not strictly necessary for the logic.
  int matx[n][3];
  /**
   * @brief Main loop to iterate through each problem.
   * It reads inputs, processes them, and updates the count in a single pass.
   */
  for (int i = 0; i < n; i++) {
    int cnt = 0; // Local counter for confident members for the current problem.
    /**
     * @brief Inner loop to read the three opinions for the i-th problem.
     */
    for (int j = 0; j < 3; j++) {
      cin >> matx[i][j];
      // Immediately check the input and update the local counter.
      if (matx[i][j] == 1)
        cnt++;
    }
    // If at least two members are confident, increment the total solution count.
    if (cnt >= 2)
      output++;
  }
  cout << output;
  return 0;
}
