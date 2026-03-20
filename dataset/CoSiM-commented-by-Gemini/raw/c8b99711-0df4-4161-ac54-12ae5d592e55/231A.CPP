/**
 * @file 231A.CPP
 * @brief Solution for Codeforces problem 231A, "Team".
 * @details This program calculates the number of problems a team of three will
 * implement, based on the rule that at least two members must be confident
 * in the solution.
 * @see http://codeforces.com/problemset/problem/231/A
 *
 * @section Algorithm
 * The program iterates through each problem and counts the number of confident members.
 *
 * 1. Read the total number of problems, `n`.
 * 2. Initialize a counter for the final result, `res`, to 0.
 * 3. Loop `n` times, once for each problem:
 *    a. For each problem, initialize a `sure` counter to 0.
 *    b. An inner loop runs 3 times to read the opinion of each team member (1 for
 *       confident, 0 for not).
 *    c. The `sure` counter is incremented for each confident member.
 *    d. After checking all three members, if the `sure` count is greater than 1,
 *       the result counter `res` is incremented.
 * 4. Finally, the total count `res` is printed.
 *
 * Note: The `notSure` variable is calculated but is redundant, as the decision
 * logic only relies on the `sure` count.
 *
 * @subsection Complexity
 * - Time Complexity: O(n), since the outer loop runs `n` times and the inner
 *   loop is constant (3 iterations).
 * - Space Complexity: O(1), as the memory usage is constant.
 */
// http://codeforces.com/problemset/problem/231/A

#include <iostream>
using namespace std;

int main() {
  // n: the number of problems.
  // t: stores the opinion of a single friend (1 or 0).
  // sure, notSure: counters for confident and non-confident opinions.
  // res: the final count of problems to be implemented.
  int n, t, sure, notSure, res = 0;
  cin >> n;
  /**
   * @brief This loop processes each of the `n` problems.
   * Pre-condition: `res` is 0. `n` holds the number of problems.
   * Invariant: After each iteration, `res` contains the count of problems
   * that satisfy the implementation criteria from the subset already evaluated.
   */
  while (n--) {
    sure = notSure = 0;
    /**
     * @brief An inner loop to poll the opinions of the three team members.
     */
    for (int i = 1; i < 4; i++) {
      cin >> t;
      if (t)
        sure++;
      else
        notSure++; // This counter is not used in the decision logic.
    }
    // If two or more members are sure, increment the result counter.
    if (sure > 1)
      res++;
  }
  cout << res;

  return 0;
}
