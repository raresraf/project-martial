/**
 * @file 158A.CPP
 * @brief Solution to a contest advancement problem (e.g., Codeforces 158A "Next Round").
 * @details This program counts the number of participants who will advance to the next round.
 * A participant advances if their score is positive and greater than or equal to the
 * score of the k-th place finisher.
 *
 * @algorithm Linear Scan
 * @complexity
 * - Time: O(n), where 'n' is the number of participants, as the code iterates through
 *   the scores twice (once for input, once for checking).
 * - Space: O(n) to store the participant scores in an array.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // n: the total number of participants.
  // k: the 1-based rank that determines the score cutoff.
  int k, n;
  cin >> n >> k;

  // A C-style variable-length array (VLA) to store the scores.
  // Note: VLAs are a non-standard C++ extension, but supported by many compilers.
  int a[n];
  /**
   * @block Input Reading
   * @logic Reads all participant scores into the array `a`.
   * The problem assumes scores are provided in non-increasing order.
   */
  for (int i = 0; i < n; i++)
    cin >> a[i];

  // The cutoff score is the score of the person at k-th place (index k-1).
  int num = a[k - 1];

  // 'ans' will store the final count of participants who advance.
  int ans = 0;
  /**
   * @block Advancement Counting
   * @logic Iterates through each score to check if it meets the advancement criteria.
   * @invariant `ans` holds the count of qualifying participants from the first `i-1` checked.
   */
  for (int i = 0; i < n; i++)
    // To advance, the score must be positive and at least the cutoff score.
    if (a[i] >= num && a[i] > 0)
      ans++;

  // Print the final count.
  cout << ans;
  return 0;
}
