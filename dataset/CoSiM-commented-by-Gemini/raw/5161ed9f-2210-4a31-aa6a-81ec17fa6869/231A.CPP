/*
Author : hardRocK
Date :
Online Judge : Codeforces
Problem Name : Team
Problem Link : http://codeforces.com/problemset/problem/231/A
*/

/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem from Codeforces.
 * @details This program determines how many problems a team of three will solve,
 * based on the rule that a problem is solved if at least two members are
 * confident about the solution.
 *
 * @algorithm
 * 1. Read the number of problems, T.
 * 2. Initialize a counter `cnt` to 0.
 * 3. Loop T times for each problem:
 *    a. Read the three members' confidence levels (a, b, c), where 1 means
 *       confident and 0 means not confident.
 *    b. Explicitly check all combinations where at least two members are
 *       confident.
 *    c. If the condition is met, increment the counter `cnt`.
 * 4. After the loop, print the total count.
 *
 * @complexity
 * Time Complexity: O(T), where T is the number of problems.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>
using namespace std;

int main() {

  int T;      // The number of problems (test cases).
  int cnt = 0; // Counter for the number of problems the team will solve.
  cin >> T;

  /**
   * @block
   * @brief Process each problem.
   */
  while (T--) {

    int a, b, c; // Confidence levels of the three members.
    cin >> a >> b >> c;

    /**
     * @brief Check if at least two members are confident.
     * @details This condition explicitly lists all pairwise combinations of
     * confident members. A simpler equivalent would be `(a + b + c >= 2)`.
     * The `(a == 1 && b == 1 && c == 1)` clause is redundant.
     */
    if ((a == 1 && b == 1) || (a == 1 && c == 1) || (b == 1 && c == 1) ||
        (a == 1 && b == 1 && c == 1)) {
      cnt++;
    }
  }

  cout << cnt << endl;
  // system("pause");
  return 0;
}
