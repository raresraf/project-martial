/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem from Codeforces.
 * @details The problem involves counting how many problems a team of three
 * will decide to implement. The team decides to implement a problem if at
 * least two of the three members are confident about the solution.
 *
 * @algorithm
 * 1. Read the total number of problems, n.
 * 2. Initialize a counter `ans` to 0, which will store the number of
 *    problems the team will solve.
 * 3. Loop n times:
 *    a. Read the confidence levels of the three members (a, b, c),
 *       represented as 0 (not confident) or 1 (confident).
 *    b. Check if the sum of their confidence levels is 2 or more.
 *    c. If it is, increment the `ans` counter.
 * 4. After the loop, print the final count `ans`.
 *
 * @complexity
 * Time Complexity: O(n), where n is the number of problems, due to the
 * single loop through all problems.
 * Space Complexity: O(1), as only a few variables are used regardless of
 * the input size.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n; // The total number of problems to consider.
  int a, b, c; // Confidence levels of the three team members (0 or 1).
  int ans = 0; // Counter for the number of problems the team will solve.
  
  cin >> n;

  /**
   * @block
   * @brief Iterate through each problem to see if the team will solve it.
   */
  for (int i = 0; i < n; i++) {
    cin >> a >> b >> c;
    /**
     * @brief Condition to check if at least two members are confident.
     * @details Since confidence is represented by 1 and lack of it by 0,
     * summing the values gives the number of confident members.
     */
    if (a + b + c >= 2) {
      ans++;
    }
  }

  // Print the total count of problems to be implemented.
  cout << ans << endl;
}
