/**
 * @file team.cpp
 * @brief Solution for Codeforces Problem 231A, "Team".
 * @details This program calculates the number of programming problems that a group of teams will solve. A team solves a problem if at least two of its three members are confident in the solution. This submission is a straightforward implementation of that logic.
 *
 * @algorithm Iterative Counting
 * @description The program reads the total number of problems, then iterates that many times. In each iteration, it reads the confidence levels of the three team members (1 for confident, 0 otherwise). If the sum of these levels is 2 or more, a counter for solved problems is incremented.
 * @complexity
 * - Time: O(N), where N is the number of problems, due to the single loop through all problems.
 * - Space: O(1), as it uses a constant number of variables.
 */

#include <iostream>
using namespace std;

int main() {
  int n, p, v, t;
  cin >> n;

  int count = 0;

  /**
   * @brief Iterates through each of the 'n' problems to be considered.
   * @invariant `count` holds the number of problems solved among the first `i` problems.
   */
  for (int i = 0; i < n; i++) {
    cin >> p >> v >> t;
    /**
     * @brief Checks if the team will solve the current problem.
     * The problem is solved if the sum of confidences (p + v + t) is greater than 1,
     * which means at least two members are confident.
     */
    if (p + v + t > 1) {
      count++;
    }
  }

  cout << count << endl;

  return 0;
}
