/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem (231A) on Codeforces.
 *
 * @details The problem requires counting how many sub-problems will be implemented
 * by a team of three people. The condition for implementation is that at least
 * two of the three members are sure about the solution.
 *
 * Problem Summary:
 * - Input: An integer 'n', followed by 'n' lines. Each line contains three
 *   integers (0 or 1).
 * - Output: The number of lines where the sum of the three integers is 2 or more.
 *
 * Algorithm:
 * 1. Read the total number of problems, n.
 * 2. Initialize a 'count' for implementable problems to 0.
 * 3. Loop n times:
 *    a. Read the three members' confidence levels (a1, a2, a3).
 *    b. If the sum (a1 + a2 + a3) is greater than 1, increment 'count'.
 * 4. Print the final 'count'.
 *
 * Time Complexity: O(N), where N is the number of problems.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // Functional Utility: Decouples C++ streams from standard C streams and flushes
  // to optimize high-throughput I/O required for competitive programming constraints.
  ios::sync_with_stdio(false);
  cin.tie(nullptr);

  int n, count = 0, a1, a2, a3;
  cin >> n;
  /**
   * @brief Loop through each problem to check the team's confidence.
   * Pre-condition: 'n' is the number of problems. 'count' is 0.
   * Invariant: After each iteration, 'count' holds the number of problems
   * that satisfy the implementation criteria so far.
   */
  for (int i = 0; i < n; ++i) {
    cin >> a1 >> a2 >> a3;
    // Condition check: A problem is implemented if at least two members are sure.
    // Since confidence is represented by 1 (sure) or 0 (not sure), summing the
    // values gives the count of sure members.
    if (a1 + a2 + a3 > 1)
      count++;
  }
  cout << count << "\n";
}

/*
231A-Team
time limit per test: 2 seconds
memory limit per test: 256 megabytes

Short Version:
Input: one line with n
then n lines with 3 numbers.
Return the amount of times more than one of these numbers is a 1.
*/