/**
 * @file 231A.CPP
 * @brief Solution to a competitive programming problem where the goal is to count
 *        the number of problems solved based on a consensus rule.
 *        Each problem has three participants, and a problem is considered solved
 *        if at least two participants agree (represented by a value of 1).
 *
 * Algorithm: Iterative problem evaluation.
 * Time Complexity: O(N) where N is the number of problems.
 * Space Complexity: O(1) as only a few variables are used.
 */

#include <iostream>
using namespace std;

/**
 * @brief Main function to execute the problem-solving logic.
 * @return 0 on successful execution.
 */
int main() {
  int n; // Declares 'n' to store the total number of problems.
  int a, b, c; // Declares 'a', 'b', 'c' to store the consensus from each participant for a problem.
  int solved = 0; // Initializes 'solved' counter to track the number of successfully solved problems.

  cin >> n; // Reads the total number of problems 'n'.

  /**
   * @brief Iterates through each problem to determine if it is solved.
   * Invariant: At the start of each iteration, 'solved' holds the count of problems
   *            determined to be solved from previous iterations.
   * Precondition: 'n' contains the number of problems to evaluate.
   */
  while (n--) {
    cin >> a >> b >> c; // Reads the consensus values for the current problem from three participants.
    
    /**
     * @brief Checks if at least two participants agree on the solution.
     * Logic: If the sum of consensus values (a + b + c) is 2 or more,
     *        it means at least two participants voted '1' (agreed).
     */
    if (a + b + c >= 2)
      solved++; // Increments the solved problem counter if the condition is met.
  }

  // Outputs the total number of problems successfully solved.
  cout << solved << endl;
  return 0;
}

