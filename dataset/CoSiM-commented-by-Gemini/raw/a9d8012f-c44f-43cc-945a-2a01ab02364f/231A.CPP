/**
 * @file 231A.CPP
 * @brief Solution for CodeForces Problem 231A - "Team".
 *
 * @details
 * The problem asks to count the number of problems that will be implemented by a team of three.
 * The team decides to implement a problem if at least two of them are sure about the solution.
 *
 * @algorithm Two-pass array-based counting
 * 1. Read the total number of problems, `n`.
 * 2. Read all `n` sets of three opinions (0 or 1) into a 2D array.
 * 3. Initialize a final counter, `result`, to 0.
 * 4. Iterate through each row (problem) of the stored inputs:
 *    a. For each problem, initialize a local counter `count` to 0.
 *    b. Iterate through the three opinions for that problem.
 *    c. If an opinion is '1', increment the local `count`.
 *    d. As soon as `count` reaches 2, increment the final `result` and break
 *       from the inner loop to move to the next problem (an optimization).
 * 5. Print the final `result`.
 *
 * This approach uses more memory than processing line-by-line by storing all inputs first.
 *
 * @time_complexity O(N), where N is the number of problems.
 * @space_complexity O(N) to store the inputs in the array (or O(1) if `n` is small
 *                   relative to the fixed array size of 1000).
 */
#include <iostream>
using namespace std;

int main() {
  int arr[1000][3]; // Array to store all inputs.
  int n; // Number of problems.
  cin >> n;

  /**
   * @brief First pass: read all inputs into the 2D array.
   */
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 3; j++) {
      cin >> arr[i][j];
    }
  }

  int count = 0;  // Local counter for 'sure' members per problem.
  int result = 0; // Final result: number of problems to be implemented.

  /**
   * @brief Second pass: iterate through stored inputs to count valid problems.
   */
  for (int i = 0; i < n; i++) {
    count = 0; // Reset for each new problem.
    for (int j = 0; j < 3; j++) {
      if (arr[i][j] == 1)
        count++;
      
      // Optimization: if we already have two 'yes' votes, we can decide and stop checking.
      if (count == 2) {
        result++;
        break; // Move to the next problem.
      }
    }
  }

  cout << result;
  return 0;
}
