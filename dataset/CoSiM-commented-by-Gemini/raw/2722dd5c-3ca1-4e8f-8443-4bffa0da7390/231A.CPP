/**
 * @file 231A.CPP
 * @brief Solution to the "Team" problem (231A) from Codeforces.
 * @details The program counts the number of problems a three-person team will solve.
 * A problem is solved if at least two out of the three members are confident
 * about the solution.
 *
 * @author Your Name (or leave blank)
 *
 * @algorithm The solution iterates through each problem and checks the team's confidence level.
 * 1. Read the total number of problems (n).
 * 2. Initialize a counter `ans` to 0, which will store the number of problems to be solved.
 * 3. Loop 'n' times, once for each problem.
 * 4. In each iteration, read the confidence levels of the three members (as 0 or 1).
 * 5. Sum the three confidence levels. If the sum is 2 or greater, it means at least
 *    two members are confident.
 * 6. If the condition is met, increment the `ans` counter.
 * 7. After the loop, print the final count `ans`.
 *
 * @complexity
 * Time Complexity: O(N), where N is the number of problems, as the program processes each
 * problem in a single pass.
 * Space Complexity: O(1), as it uses a fixed amount of memory.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, i, a, b, c, ans = 0;
  cin >> n;
  /**
   * @block
   * @brief Processes each of the 'n' problems.
   * @invariant 'ans' holds the count of problems solved so far.
   * @pre 'n' is the total number of problems.
   * @post 'ans' holds the total count of problems where at least two members are confident.
   */
  for (i = 0; i < n; i++) {
    cin >> a >> b >> c;
    // A problem is implemented if the sum of confidences (0 or 1) is at least 2.
    if (a + b + c >= 2) {
      ans++;
    }
  }
  cout << ans << endl;
  return 0;
}
