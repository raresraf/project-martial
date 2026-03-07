/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem from Codeforces.
 *
 * @details
 * This program calculates the number of problems a three-person team will solve,
 * based on the condition that a problem is solved only if at least two members
 * are confident in the solution. This is a space-efficient implementation that
 * uses C-style I/O.
 *
 * Algorithm:
 * 1. Read the total number of problems, `t`.
 * 2. Initialize a final counter for solved problems (`count2`) to 0.
 * 3. Loop `t` times for each problem:
 *    a. Initialize a temporary counter for the current problem (`count1`) to 0.
 *    b. In an inner loop, read the three confidence levels (0 or 1).
 *    c. For each confidence level, if it is 1, increment `count1`.
 *    d. After reading the three inputs, if `count1` is 2 or greater, increment
 *       the final counter `count2`.
 *    e. Reset `count1` to 0 for the next problem.
 * 4. Print the final value of `count2`.
 *
 * Time Complexity: O(N), where N is the number of problems.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  int t, a, count1 = 0, count2 = 0;
  // t: number of problems
  // a: temporary storage for one team member's opinion
  // count1: temporary counter for confident members per problem
  // count2: final counter for total solved problems
  scanf("%d", &t);
  /**
   * @brief Outer loop to iterate through each of the 't' problems.
   */
  for (int i = 1; i <= t; i++) {
    /**
     * @brief Inner loop to read the opinions of the three team members.
     */
    for (int j = 1; j <= 3; j++) {
      scanf("%d", &a);
      if (a == 1) {
        count1++;
      }
    }
    // If two or more members are confident, the team solves the problem.
    if (count1 >= 2) {
      count2++;
    }
    // Reset the temporary counter for the next problem.
    count1 = 0;
  }
  printf("%d\n", count2);

  return 0;
}
