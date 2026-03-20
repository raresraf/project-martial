/**
 * @file 231A.CPP
 * @brief Solution for the "Team" problem.
 * @details The problem is to count how many sub-problems a team of three members
 * will decide to implement. The team implements a problem if at least two out
 * of the three members are sure about its solution.
 *
 * Algorithm:
 * 1. Read the total number of sub-problems, 'n'.
 * 2. Initialize a counter for implementable problems to zero.
 * 3. Loop 'n' times, once for each sub-problem:
 *    a. Read the confidence levels of the three members (1 for sure, 0 for not).
 *    b. Check if the sum of their confidence levels is 2 or more.
 *    c. If it is, increment the counter.
 * 4. Print the final count.
 *
 * Time Complexity: O(N), where N is the number of sub-problems.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  /**
   * Functional Utility: Decouples C++ streams from C stdio and unties cin from cout
   * to significantly speed up I/O, a common practice in competitive programming.
   */
  ios::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(NULL);

  short n, a, b, c, cnt = 0;

  cin >> n;
  /**
   * Block Logic: Process each sub-problem to see if it will be implemented.
   * Invariant: The loop runs 'n' times. `cnt` holds the count of problems to
   * implement from the previous iterations.
   */
  while (n--) {
    cin >> a >> b >> c;
    /**
     * Core Logic: Check if at least two members are sure.
     * In C++, non-zero integers are treated as `true`. This condition checks
     * if any pair of members is confident. An equivalent and common simplification
     * for this logic is `if (a + b + c >= 2)`.
     */
    if ((a && b) || (b && c) || (c && a))
      cnt++;
  }
  cout << cnt << "
";
}
