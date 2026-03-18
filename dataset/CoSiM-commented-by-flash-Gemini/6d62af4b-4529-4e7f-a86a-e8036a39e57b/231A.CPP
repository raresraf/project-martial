/**
 * @file 231A.CPP
 * @brief This program counts the number of problem solutions based on a simple majority rule.
 * For each of 'n' problems, three friends decide whether they are sure about the solution.
 * A problem is considered implemented if at least two friends are sure.
 * Algorithm: Iterative processing of 'n' sets of decisions, summing 'sure' votes, and incrementing a counter
 * if the sum meets the criteria.
 * Time Complexity: O(N) where 'N' is the number of problems.
 * Space Complexity: O(1) as only a few variables are used to store state.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int n;
  cin >> n;
  int x, y, z, sum = 0, count = 0;
  /**
   * @brief Iterates 'n' times to process the decisions for each problem.
   * Pre-condition: 'n' is the total number of problems. 'count' is initialized to 0.
   * Invariant: In each iteration, 'x', 'y', 'z' represent the decisions (0 or 1) of three friends for the current problem.
   */
  for (int i = 0; i < n; i++) {
    cin >> x >> y >> z;
    sum = x + y + z;
    /**
     * @brief Checks if at least two friends are sure about the solution for the current problem.
     * Logic: If the sum of decisions (x + y + z) is 2 or more, the problem is considered implemented.
     * Post-condition: If the condition is met, 'count' is incremented.
     */
    if (sum >= 2)
      count++;
  }
  // Outputs the total number of problems that were considered implemented.
  cout << count << endl;
  return 0;
}