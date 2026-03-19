/**
 * @file 231A.CPP
 * @brief Solution for the "Team" problem (Codeforces 231A).
 * @details This program calculates the number of problems a team of three will
 * implement, based on the condition that at least two members must be confident
 * about the solution.
 *
 * Algorithm: Summation and Threshold Check
 * This implementation uses a clear and direct approach. For each problem, it reads
 * the three confidence values (1 for sure, 0 for not sure) and calculates their sum.
 * If the sum is 2 or greater, it means at least two members are confident, and a
 * counter for implemented problems is incremented.
 *
 * Time Complexity: O(N), where N is the number of problems, as it iterates once
 * through the list of problems.
 * Space Complexity: O(1), as it uses a constant amount of memory.
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  // n: the number of problems to evaluate.
  int n;
  cin >> n;

  // total: a counter for the number of problems that will be implemented.
  int total = 0;

  /**
   * Block Logic: Iterate through each of the 'n' problems.
   */
  while (n--) {
    // a, b, c: confidence levels of the three team members.
    int a, b, c;
    // Note: Mixing cin and scanf is generally safe here but can require
    // synchronization in more complex scenarios.
    scanf("%d %d %d", &a, &b, &c);

    // Calculate the sum of confidence votes.
    int sum = a + b + c;

    // Core Logic: If two or more members are sure, the team will implement the problem.
    if (sum >= 2) {
      total++;
    }
  }
  
  // Output the final count.
  cout << total << endl;

  return 0;
}