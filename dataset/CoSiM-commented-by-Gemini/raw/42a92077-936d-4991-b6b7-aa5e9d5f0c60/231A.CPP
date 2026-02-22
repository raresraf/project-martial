/**
 * @file 231A.CPP
 * @brief Solution to Codeforces problem 231A - Team.
 *
 * @algorithm
 * The problem is to count the number of problems a three-person team will solve.
 * The team decides to solve a problem if at least two of the three members are
 * sure about the solution.
 *
 * The algorithm is as follows:
 * 1. Read the total number of problems, `n`.
 * 2. Initialize a counter for solved problems, `sum`, to 0.
 * 3. Loop `n` times, once for each problem.
 * 4. In each iteration, read the three members' confidence levels (0 for unsure,
 *    1 for sure) into variables `a`, `b`, and `c`.
 * 5. Check if the sum of `a`, `b`, and `c` is greater than or equal to 2.
 *    This is a simple and efficient way to determine if at least two members
 *    are sure.
 * 6. If the condition is met, increment the `sum` counter.
 * 7. After the loop, print the total `sum`.
 *
 * @complexity
 * - Time Complexity: O(N), where N is the number of problems, as the code
 *   iterates through each problem once.
 * - Space Complexity: O(1), as it only uses a few variables regardless of N.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  /**
   * @brief Fast I/O optimization for C++.
   * `ios::sync_with_stdio(false)` unties C++ streams from C streams.
   * `cin.tie(0)` or `cin.tie(nullptr)` prevents `cin` from flushing `cout`
   * before each input operation.
   */
  ios::sync_with_stdio(false);
  cin.tie(0);

  int n, sum(0), a, b, c;
  cin >> n;

  /**
   * @brief This loop processes each of the `n` problems.
   */
  while (n--) {
    cin >> a >> b >> c;

    /**
     * @brief Checks if at least two members are sure.
     * Since sure=1 and not_sure=0, the sum will be >= 2 if and only
     * if two or three members are sure.
     */
    if (a + b + c >= 2)
      sum++;
  }
  cout << sum << "\n";
}
