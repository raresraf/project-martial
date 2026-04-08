/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem from Codeforces (Problem 231A).
 * @details This program counts how many problems a team of three will solve.
 * The condition for solving a problem is that at least two members are
 * confident about the solution.
 *
 * @algorithm
 * 1. Read the total number of problems, `t`.
 * 2. Initialize a total counter for solved problems (`count`) to zero.
 * 3. Loop `t` times, once for each problem.
 *    a. Inside the loop, initialize a temporary counter `c` for the current problem to zero.
 *    b. Loop 3 times to read the confidence level (0 or 1) of each of the three members.
 *    c. If a member is confident (input is 1), increment the temporary counter `c`.
 *    d. After checking all three members, if `c` is 2 or greater, increment the total counter `count`.
 * 4. After the main loop finishes, print the total `count`.
 *
 * @time_complexity O(t), where `t` is the number of problems (since the inner loop is constant 3).
 * @space_complexity O(1).
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int t; // Total number of problems.
  int c; // Temporary counter for confident members in a single problem.
  int a; // Stores the confidence level of a single member.
  cin >> t;
  int count = 0; // Total count of problems the team will solve.

  /**
   * @block_logic Outer loop iterates through each problem.
   */
  for (int i = 0; i < t; ++i) {
    c = 0;
    /**
     * @block_logic Inner loop reads the confidence of the three members
     * and counts how many are sure.
     */
    for (int i = 0; i < 3; ++i) {
      cin >> a;
      if (a == 1)
        c++;
    }
    // If two or more members are confident, the team solves the problem.
    if (c >= 2)
      count++;
  }

  cout << count;
}