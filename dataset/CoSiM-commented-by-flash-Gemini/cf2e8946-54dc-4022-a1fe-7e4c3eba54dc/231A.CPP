/**
 * @file 231A.CPP
 * @brief Determines the number of problems a team can solve based on individual confidence.
 *   This program simulates a scenario where three friends decide whether to implement
 *   a problem's solution based on their confidence (1 if sure, 0 if not sure).
 *   A problem is implemented if at least two friends are sure about the solution.
 * * Algorithm: Majority Count.
 *   For each problem, the program counts how many friends are sure about the solution.
 *   If this count is 2 or more, the problem is considered solved, and a counter is incremented.
 * Time Complexity: O(N) where N is the number of problems. Each problem involves a constant
 *   number of operations (reading 3 integers, 3 comparisons, 1 addition, 1 comparison).
 * Space Complexity: O(1) as only a few variables are used (array 'a' has fixed small size).
 */

#include <iostream>
using namespace std;

/**
 * @brief Main function to calculate the total number of problems the team will implement.
 * Functional Utility: Handles input, processes confidence levels for each problem, and aggregates the results.
 */
int main() {
  int n; // Number of problems to consider.
  // 'a' stores the confidence levels for the current problem (fixed size 5, but only 3 used).
  // 'flag' counts the number of problems the team will solve.
  int a[5], flag = 0;
  cin >> n; // Reads the number of problems.

  /**
   * Block Logic: Iterates through each problem, evaluating the team's ability to solve it.
   * Invariant: At the start of each outer iteration, 'i' represents the current problem index,
   *   and 'flag' holds the count of problems solved so far.
   */
  for (int i = 0; i < n; i++) {
    /**
     * Block Logic: Reads the confidence levels for the current problem from three friends.
     * Invariant: After this loop, 'a[0]', 'a[1]', 'a[2]' contain the confidence levels.
     */
    for (int j = 0; j < 3; j++) {
      cin >> a[j];
    }
    int count = 0; // Counter for friends who are sure about the solution for the current problem.
    /**
     * Block Logic: Checks each friend's confidence level for the current problem.
     * Invariant: 'count' accumulates the number of confident friends for the current problem.
     */
    for (int j = 0; j < 3; j++) {
      /**
       * Condition: If a friend is confident (value is 1), increment the counter.
       */
      if (a[j] == 1) {
        count++;
        /**
         * Condition: If at least two friends are confident, the problem can be solved.
         * Increment 'flag' and break from the inner loop as the decision for this problem is made.
         */
        if (count >= 2) {
          flag++;
          break;
        }
      }
    }
  }
  cout << flag; // Outputs the total number of problems the team will implement.
  return 0;     // Indicates successful program execution.
}
