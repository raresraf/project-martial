/**
 * @file 231A.CPP
 * @brief Solution for Codeforces Problem 231A, "Team".
 * @details This program calculates the number of problems that will be implemented by a three-person team.
 * The team decides to implement a problem if at least two of its members are sure about the solution.
 *
 * Algorithm:
 * 1. Read the total number of problems, `n`.
 * 2. Initialize a counter for implementable problems to zero.
 * 3. Loop `n` times, once for each problem.
 * 4. In each iteration, read the confidence of the three members (1 for sure, 0 for not sure).
 * 5. If the sum of their confidences is 2 or more, increment the counter.
 * 6. After the loop, print the final count.
 *
 * Time Complexity: O(N), where N is the number of problems.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>
using namespace std;
#define ll long long

int main() {
  /**
   * @brief Optimizes C++ stream I/O for faster execution in competitive programming.
   * `ios_base::sync_with_stdio(false)` disables synchronization with C-style I/O.
   * `cin.tie(NULL)` unties cin from cout.
   */
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  int n, a, b, c;
  cin >> n;
  int res = 0;
  /**
   * @brief Iterates through each problem to determine if it will be implemented.
   */
  for (int i = 0; i < n; i++) {
    cin >> a >> b >> c;
    /**
     * @brief Checks if at least two team members are sure about the solution.
     * The condition `a + b + c > 1` is equivalent to `a + b + c >= 2`.
     */
    if (a + b + c > 1)
      res++;
  }
  cout << res;
}
