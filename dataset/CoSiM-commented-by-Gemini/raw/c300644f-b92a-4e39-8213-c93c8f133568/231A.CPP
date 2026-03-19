/**
 * @file 231A.CPP
 * @brief Solution for the "Team" problem (Codeforces 231A).
 * @details This program calculates the number of problems a team of three will
 * implement. The condition for implementation is that at least two of the three
 * members are confident about the solution.
 *
 * Algorithm: Majority Vote Counting
 * The program iterates through each problem. For each problem, it reads the
 * confidence level (1 for sure, 0 for not sure) of the three team members.
 * It then checks if the sum of their confidence levels is 2 or greater.
 * The implementation uses a clever bitwise trick `(a & b || b & c || c & a)`
 * to check if at least two members are sure, which is equivalent to checking
 * if `a+b+c >= 2`. A counter is incremented for each problem that meets this criterion.
 *
 * Time Complexity: O(N), where N is the number of problems, due to the single loop.
 * Space Complexity: O(1), as it uses a fixed amount of memory.
 */

#include <bits/stdc++.h>

using namespace std;

int main() {

  // n: the total number of problems to consider.
  int n;
  // a, b, c: the confidence levels (1 or 0) of the three team members for a single problem.
  int a, b, c;
  // count: the number of problems that will be implemented.
  int count = 0;

  scanf("%d", &n);

  /**
   * Block Logic: Iterate through each problem to see if it will be implemented.
   * Pre-condition: 'n' holds the number of problems.
   */
  while (n--) {

    scanf("%d%d%d", &a, &b, &c);

    /**
     * Core Logic: Check if at least two members are sure.
     * The expression `(a & b || b & c || c & a)` evaluates to true if any pair
     * of members is sure (i.e., their confidence is 1). This is a bitwise
     * method to achieve the same result as `(a + b + c >= 2)`.
     */
    if (a & b || b & c || c & a)
      count++;
  }

  // Output the total count of problems to be implemented.
  cout << count << endl;
}
