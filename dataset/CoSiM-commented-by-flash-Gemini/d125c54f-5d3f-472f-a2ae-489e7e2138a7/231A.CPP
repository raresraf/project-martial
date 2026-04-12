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
 * Space Complexity: O(1) as only a few variables are used to store counts and temporary inputs.
 */

#include <iostream>
using namespace std;

/**
 * @brief Main function to calculate the total number of problems the team will implement.
 * Functional Utility: Handles input, processes confidence levels for each problem, and aggregates the results.
 */
int main() {
  int n; // Number of problems to consider.
  cin >> n;

  int answer = 0; // Counter for the number of problems the team will solve.

  /**
   * Block Logic: Iterates through each problem, evaluating the team's ability to solve it.
   * Invariant: At the start of each iteration, 'i' represents the current problem index,
   *   and 'answer' holds the count of problems solved so far.
   */
  for (int i = 0; i < n; i++) {
    int a1, a2, a3; // Confidence levels of the three friends for the current problem.
    cin >> a1 >> a2 >> a3;

    int num_1 = 0; // Counter for friends who are sure about the solution (value 1).

    /**
     * Block Logic: Checks each friend's confidence level.
     * Invariant: 'num_1' accumulates the count of confident friends.
     */
    if (a1 == 1) {
      num_1++;
    }
    if (a2 == 1) {
      num_1++;
    }
    if (a3 == 1) {
      num_1++;
    }

    /**
     * Condition: Determines if the team will implement the problem.
     * If two or more friends are sure, the problem is added to the solved count.
     */
    if (num_1 >= 2) {
      answer++;
    }
  }
  cout << answer; // Outputs the total number of problems the team will implement.
}