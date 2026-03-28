/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem from Codeforces.
 *
 * This program counts the number of problems a team will implement, given that
 * a problem is implemented only if at least two of the three members agree on
 * a solution.
 *
 * @algorithm The code uses nested loops. The outer loop iterates through each
 * problem. The inner loop iterates three times to read the opinion of each team
 * member. A counter `c` tracks how many members are sure for the current problem.
 * When `c` reaches exactly 2, the final answer `d` is incremented. This logic
 * works because the count for a single problem can only pass through the state `c=2`
 * once.
 *
 * @time_complexity O(N), as the inner loop is a fixed size (3) for each of N problems.
 * @space_complexity O(1).
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, i, x, c = 0, d = 0, j;
  cin >> n;
  /**
   * @block_logic Outer loop iterates through each of the `n` problems.
   */
  for (j = 1; j <= n; j++) {
    c = 0; // Counter for "sure" members for the current problem.
    /**
     * @block_logic Inner loop reads the three opinions for the current problem.
     */
    for (i = 1; i <= 3; i++) {
      cin >> x;
      if (x == 1) {
        c++;
        /**
         * @invariant This condition increments the final answer `d` as soon as
         * the second "sure" member is found. Because it checks for `c == 2`
         * and not `c >= 2`, it will only increment `d` once per problem.
         */
        if (c == 2) {
          d++;
          // break;
          // cout<<"d="<<d<<endl;
        }
      }
    }
  }
  cout << d << endl;
  return 0;
}
