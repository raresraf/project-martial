/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem from Codeforces (231A).
 * @details The program calculates the number of problems a team of three will solve.
 * A problem is solved if at least two members are confident in the solution.
 *
 * Algorithm: Simple Counting.
 * 1. Read the total number of problems, `n`.
 * 2. Loop `n` times, once for each problem.
 * 3. For each problem, read the confidence levels (0 or 1) of the three members.
 * 4. Sum the confidence levels. If the sum is 2 or greater, it means at least
 *    two members are confident, so increment a solution counter.
 * 5. After the loop, print the total count of solvable problems.
 *
 * Time Complexity: O(N) where N is the number of problems.
 * Space Complexity: O(1) as storage is constant regardless of input size.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, ans = 0, p, v, t;
  // Read the number of problems to evaluate.
  scanf("%d", &n);
  /**
   * Block Logic: Iterate through each problem to determine if it will be solved.
   * Pre-condition: `n` contains the total number of problems.
   */
  for (int i = 0; i < n; i++) {
    // Read the confidence levels for Petya (p), Vasya (v), and Tonya (t).
    scanf("%d %d %d", &p, &v, &t);
    /**
     * Block Logic: Check if the problem meets the solution criteria.
     * The problem is solved if the sum of confidences is at least 2.
     */
    if (p + v + t > 1)
      ans++;
  }
  // Print the total number of problems that will be solved.
  printf("%d", ans);
  return 0;
}