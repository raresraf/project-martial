/**
 * @file 231A.CPP
 * @brief Solution for Codeforces Problem 231A, "Team".
 * @details This program calculates the number of problems that will be solved by a group of teams. A team decides to solve a problem only if at least two of its three members are confident about the solution.
 *
 * @algorithm Iterative Counting
 * @description The program reads the number of problems, then iterates through each one. For each problem, it sums up the confidence levels of the three team members (represented by 1 for confident, 0 for not). If the sum is 2 or greater, it means the team will solve the problem, and a counter is incremented.
 * @complexity
 * - Time: O(N), where N is the number of problems, as it processes each problem in a single loop.
 * - Space: O(1), using a fixed amount of memory regardless of the input size.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, p, v, t, count = 0;
  cin >> n;

  /**
   * @brief This loop iterates through each of the 'n' problems.
   * @invariant `count` holds the number of problems solved by the teams among the first `i` problems.
   */
  while (n--) {
    cin >> p >> v >> t;
    
    /**
     * @brief Checks if the team will solve the current problem.
     * The condition `(p + v + t) > 1` is a concise way to check if at least two members are sure (i.e., if the sum of their 1s and 0s is 2 or 3).
     */
    if ((p + v + t) > 1)
      count++;
  }

  // Output the total number of problems that will be solved.
  cout << count << endl;
}
