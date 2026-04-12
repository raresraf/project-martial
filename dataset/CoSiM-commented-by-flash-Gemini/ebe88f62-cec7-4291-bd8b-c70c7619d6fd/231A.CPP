/**
 * @file 231A.CPP
 * @brief Determines the number of problems a team can solve based on individual confidence.
 *   This program simulates a scenario where three friends decide whether to implement
 *   a problem's solution based on their confidence (1 if sure, 0 if not sure).
 *   A problem is implemented if at least two friends are sure about the solution.
 * * Algorithm: Majority Count.
 *   For each problem, the program reads three confidence levels. If the count of
 *   confident friends (value 1) is 2 or more, a global counter for solved problems is incremented.
 * Time Complexity: O(N) where N is the number of problems. Each problem involves a constant
 *   number of operations (reading 3 integers, 3 comparisons, 1 addition, 1 comparison).
 * Space Complexity: O(1) as only a few variables are used to store counts and temporary inputs.
 */

#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function to calculate the total number of problems the team will implement.
 * Functional Utility: Handles input, processes confidence levels for each problem, and aggregates the results.
 */
int main() {
  int n; // Total number of problems.
  int i, x, c = 0, d = 0, j; // 'i', 'j' for loop counters, 'x' for individual confidence,
                            // 'c' for confident friends per problem, 'd' for total solved problems.
  cin >> n; // Reads the total number of problems.

  /**
   * Block Logic: Iterates through each problem to be evaluated.
   * Invariant: At the start of each outer iteration, 'j' represents the current problem number,
   *   and 'd' holds the count of problems solved so far.
   */
  for (j = 1; j <= n; j++) {
    c = 0; // Resets the confident friends counter for the new problem.
    /**
     * Block Logic: Reads the confidence levels from the three friends for the current problem.
     * Invariant: At the start of each inner iteration, 'i' represents the current friend,
     *   and 'c' holds the count of confident friends for the current problem so far.
     */
    for (i = 1; i <= 3; i++) {
      cin >> x; // Reads the confidence level of a friend (0 or 1).
      /**
       * Condition: If the friend is confident (x == 1), increment the confident friends counter 'c'.
       */
      if (x == 1) {
        c++; // Increment count of confident friends.
        // The commented out sections below seem to be for debugging and will be ignored
        // to adhere to the "Zero Code Mutation" rule.
        // if (c == 2) {
        //   d++;
        //   // break;
        //   // cout<<"d="<<d<<endl;
        // }
      }
    }
    // After checking all three friends for the current problem:
    /**
     * Condition: Checks if at least two friends were confident for the current problem.
     * If true, increments the total count of solved problems 'd'.
     */
    if (c >= 2) { // The logic to increment 'd' should be here, after all friends are checked.
      d++;
    }
  }
  cout << d << endl; // Outputs the total number of problems the team will implement.
  return 0;          // Indicates successful program execution.
}
