/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem from Codeforces.
 * @details This program determines how many problems a team of three will implement.
 * A problem is implemented if at least two of the three members are confident
 * in their solution.
 *
 * @section Algorithm
 * The program first reads an integer `n`, which is the total number of problems.
 * It then enters a loop that iterates `n` times. In each iteration, it reads
 * three integers (0 or 1), which represent the confidence vote of each team member
 * for the current problem. If the sum of these three votes is greater than 1
 * (i.e., at least 2), a counter is incremented. After the loop finishes, the
 * final value of the counter is printed, which represents the total number of
 * problems that will be implemented.
 *
 * @section Complexity
 * - Time Complexity: O(n), where 'n' is the number of problems, because the program
 *   iterates through each problem once.
 * - Space Complexity: O(1), as the memory usage is constant regardless of the
 *   number of problems.
 */

#include <iostream>

using namespace std;

int main() {
  /**
   * @var n The total number of problems to evaluate.
   * @var a, b, c The confidence votes (0 or 1) of the three team members.
   * @var counter Accumulates the count of problems that will be implemented.
   */
  int n, a, b, c;
  int counter = 0;

  // Read the total number of problems from standard input.
  cin >> n;

  /**
   * @brief Block Logic: Iterates through each of the 'n' problems to process the votes.
   * @invariant At the beginning of each iteration, `counter` holds the number of
   * problems approved from the previous iterations.
   */
  for (int i = 0; i < n; i++) {
    // Read the three votes for the current problem.
    cin >> a >> b >> c;

    /**
     * @brief Logic: Checks if the team will implement the problem.
     * The condition `a + b + c > 1` is true if two or three members are confident
     * (voted '1').
     */
    if (a + b + c > 1)
      counter++;
  }

  // Output the total number of problems that will be implemented.
  cout << counter << endl;
  return 0;
}