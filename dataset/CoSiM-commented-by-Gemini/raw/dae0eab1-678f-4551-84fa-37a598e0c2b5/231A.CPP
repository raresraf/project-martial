/**
 * @file 231A.CPP
 * @brief Solution for Codeforces Problem 231A - "Team".
 * @details This program counts how many problems a team will solve, based on the rule
 * that a problem is solved if at least two of the three members are confident.
 * This implementation uses C-style I/O for performance.
 *
 * @author (unknown)
 *
 * @algorithm
 * 1. An integer `t` representing the number of problems is read from standard input.
 * 2. A `sum` variable is initialized to 0 to count the solvable problems.
 * 3. A `for` loop is used to iterate `t` times. The loop reads `t` in its
 *    initialization statement and decrements it in the post-expression (`t--`).
 * 4. Inside the loop, the confidence levels of the three members (`a`, `b`, `c`) are read.
 * 5. If the sum of their confidences is greater than 1 (i.e., at least 2), the `sum` counter is incremented.
 * 6. After the loop, the final `sum` is printed to standard output.
 *
 * @complexity
 * Time Complexity: O(t), where t is the number of problems.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int t, a, b, c, sum = 0;
  /**
   * Block Logic: This compact `for` loop reads the number of test cases `t` and
   * iterates `t` times, processing one problem in each iteration.
   */
  for (scanf("%d", &t); t--;) {
    // Read the three members' confidence levels using C-style input.
    scanf("%d %d %d", &a, &b, &c);
    /**
     * Block Logic: If the sum of confidence levels is 2 or more, the problem is solvable.
     */
    if (a + b + c > 1)
      sum++;
  }
  // Print the total count of solvable problems using C-style output.
  printf("%d\n", sum);
  return 0;
}
