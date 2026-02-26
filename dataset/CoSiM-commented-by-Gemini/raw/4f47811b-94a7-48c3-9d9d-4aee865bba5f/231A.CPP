/**
 * @file 231A.CPP
 * @brief A clean, space-efficient solution to the "Team" problem.
 * @details The program counts how many problems a three-person team will solve.
 * A problem is solved if at least two of the three members are confident in the
 * solution (represented by 1).
 *
 * @algorithm
 * This solution is optimized for space by processing each problem's data as it
 * is read, without storing the inputs for all problems.
 * 1. Read the number of problems, `n`.
 * 2. Loop `n` times, once for each problem.
 * 3. Inside the loop, use a small, temporary array to store the three opinions
 *    for the current problem.
 * 4. Sum the three opinions. If the sum is 2 or greater, increment a result
 *    counter.
 * 5. After the loop finishes, print the final result.
 *
 * @complexity
 * Time Complexity: O(N), where N is the number of problems, for a single pass.
 * Space Complexity: O(1), as it only uses a small, constant-size array within
 * the loop.
 */

#include <iostream>

using namespace std;

int main() {
  int n = 0; // Number of problems to process.
  cin >> n;
  int solvable_problems_count = 0; // Counter for the final result.

  /**
   * @block
   * @brief Main loop to process each of the `n` problems.
   */
  for (int i = 0; i < n; i++) {
    // A temporary local array to hold the 3 opinions for the current problem.
    int nums[3];

    cin >> nums[0];
    cin >> nums[1];
    cin >> nums[2];

    /**
     * @block
     * @brief Check if the current problem will be solved.
     * @details The sum of the three opinions (0 or 1) gives the number of
     * members who are confident.
     */
    if (nums[0] + nums[1] + nums[2] >= 2) {
      solvable_problems_count++;
    }
  }

  // Print the total number of problems that will be solved.
  cout << solvable_problems_count;
  return 0;
}