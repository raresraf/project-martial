/**
 * @file 231A.CPP
 * @brief Solution for the "Team" problem (e.g., Codeforces 231A).
 * @details This program calculates the number of problems a team will implement.
 * A problem is implemented if at least two of the three members are confident.
 * This solution uses a two-pass approach, first storing the confidence counts
 * for each problem and then iterating through them to get a final answer.
 *
 * @algorithm Two-pass iterative counting.
 * @complexity
 * - Time: O(n), where 'n' is the number of problems. The code iterates through
 *   the problems twice in separate, non-nested loops.
 * - Space: O(n) to store the confidence counts in a vector.
 *
 * @note This approach is unnecessarily complex. A more efficient O(1) space solution
 * could be achieved in a single pass by checking the confidence count and incrementing
 * the final answer directly within the first loop, removing the need for the `cnt` vector.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // t: the number of problems.
  int t;
  cin >> t;
  // cnt: a vector to store the number of "sure" members for each problem.
  vector<int> cnt(t);
  // ans: the final count of problems to be implemented.
  int ans = 0;

  /**
   * @block First Pass: Count and Store Confidence
   * @logic This loop reads the data for each problem and stores the number of
   * "sure" members (1s) in the `cnt` vector.
   */
  while (t--) {
    int ct = 0; // Local counter for the current problem.
    for (int i = 0; i < 3; i++) {
      int q;
      cin >> q;
      if (q == 1)
        ct++;
    }
    // NOTE: Because the `while(t--)` loop counts `t` down from t-1 to 0,
    // the results are stored in the vector in reverse order.
    cnt[t] = ct;
  }

  /**
   * @block Second Pass: Tally Final Answer
   * @logic Iterates through the stored confidence counts and increments the
   * final answer if a count is 2 or more.
   */
  for (auto i : cnt)
    if (i >= 2)
      ans++;

  cout << ans << endl;
  return 0;
}
