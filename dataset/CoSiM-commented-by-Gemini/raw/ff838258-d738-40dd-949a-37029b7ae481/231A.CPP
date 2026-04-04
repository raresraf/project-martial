/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem (231A) on Codeforces.
 * @author chavodon
 *
 * @details This program counts the number of problems a team of three will solve.
 * A problem is solved if at least two out of the three members are confident
 * in their solution.
 *
 * Link: https://codeforces.com/problemset/problem/231/A
 *
 * Algorithm:
 * 1. Read the total number of problems, 'n'.
 * 2. Initialize a counter 'sum' to 0.
 * 3. Loop 'n' times:
 *    a. Read the confidence levels of the three members (n1, n2, n3), where
 *       1 means confident and 0 means not.
 *    b. If the sum of the confidence levels is 2 or greater, increment the 'sum' counter.
 * 4. Print the final 'sum'.
 *
 * Time Complexity: O(N), where N is the number of problems.
 * Space Complexity: O(1).
 */
#include <iostream>

using namespace std;

int main() {
  int n, i, n1, n2, n3, sum = 0;
  cin >> n;
  /**
   * @brief Loop through each problem to check the team's confidence.
   * Pre-condition: 'n' is the number of problems. 'sum' is 0.
   * Invariant: After each iteration, 'sum' holds the count of problems
   * that meet the criteria to be solved.
   */
  for (i = 0; i < n; i++) {
    cin >> n1 >> n2 >> n3;
    // Check if at least two members are confident.
    if ((n1 + n2 + n3) >= 2)
      sum++;
  }
  cout << sum;
  return 0;
}