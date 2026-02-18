/**
 * @file 546A.CPP
 * @brief Solution to a problem involving finding remaining intervals.
 * @details This program reads `n` intervals `[l, r]` and an integer `k`. It finds the first
 * interval that contains `k` and outputs how many intervals are remaining in the list from
 * that point onward, including the one found.
 *
 * @note The filename `546A.CPP` appears to be incorrect, as the logic does not match
 * the "Soldier and Bananas" problem (546A). The logic solves a different problem as described.
 *
 * @author czqsole (as per original comment)
 *
 * @algorithm The solution uses a simple linear search.
 * 1. Read the number of intervals, `n`.
 * 2. Read `n` pairs of integers, representing the start and end of each interval.
 * 3. Read the integer `k` to be checked.
 * 4. Iterate through the intervals from the beginning.
 * 5. For each interval `i`, check if `k` falls within `[l[i], r[i]]`.
 * 6. If `k` is in the interval, break the loop. The current index `i` is the first
 *    chapter that contains the page.
 * 7. The number of chapters remaining is `n - i`. Print this result.
 *
 * @complexity
 * Time Complexity: O(N), where N is the number of intervals, due to the single pass search.
 * Space Complexity: O(N), to store the `l` and `r` arrays for the intervals.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, k;
  // Arrays to hold the start and end points of the intervals.
  int l[105] = {0};
  int r[105] = {0};
  cin >> n;
  /**
   * @block
   * @brief Reads `n` intervals from standard input.
   */
  for (int i = 0; i < n; i++) {
    cin >> l[i] >> r[i];
  }
  cin >> k;
  int i = 0;
  /**
   * @block
   * @brief Finds the index of the first interval containing 'k'.
   * @invariant The intervals from 0 to `i-1` do not contain `k`.
   * @post `i` is the index of the first interval `[l[i], r[i]]` such that `l[i] <= k <= r[i]`.
   */
  for (; i < n; i++) {
    if (l[i] <= k && k <= r[i]) {
      break;
    }
  }
  // The number of remaining chapters is the total count `n` minus the index `i` of the current chapter.
  int ans = n - i;
  cout << ans << endl;
}