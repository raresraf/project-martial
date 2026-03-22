/**
 * @file 266A.CPP
 * @brief Solves the "Stones on the Table" problem (Codeforces 266A).
 * @details This program calculates the minimum number of stones to remove from a
 * row so that no two adjacent stones share the same color.
 *
 * @section algorithm
 * Algorithm: Single-Pass Counting
 * The core insight is that to eliminate an adjacent pair of same-colored stones,
 * one of them must be removed. The minimum number of removals is therefore
 * exactly the number of such adjacent pairs.
 * The code implements this by iterating through the string of stones once,
 * comparing each stone to its predecessor. If they are the same, a counter
 * is incremented.
 *
 * @section complexity
 * - Time Complexity: O(N), where N is the number of stones, as the solution
 *   involves a single pass through the input string.
 * - Space Complexity: O(N), to store the string representing the stones.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, len_s;
  int result = 0;
  string s;
  cin >> n;
  cin >> s;

  /**
   * @brief This loop iterates through the stones to find adjacent pairs of the same color.
   * It starts from the second stone (index 1) and compares each stone to the one
   * immediately before it.
   */
  for (int i = 1; i < n; i++) {
    /**
     * @brief Checks for an adjacent pair of same-colored stones.
     * If found, one must be removed, so the result counter is incremented.
     */
    if (s[i - 1] == s[i]) {
      result++;
    }
  }

  // The final result is the total count of stones that need to be removed.
  cout << result;
  return 0;
}
