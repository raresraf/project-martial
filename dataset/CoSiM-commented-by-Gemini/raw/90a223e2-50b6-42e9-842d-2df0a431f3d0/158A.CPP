/**
 * @file 158A.CPP
 * @brief Solution for the "Next Round" problem (e.g., Codeforces 158A).
 * @details This program determines the number of participants who will advance to the
 * next round. A participant advances if they have a positive score and their score
 * is greater than or equal to the score of the k-th place finisher.
 *
 * @algorithm Linear Scan
 * @complexity
 * - Time: O(n), where 'n' is the number of participants, due to the loops for
 *   reading input and counting qualifiers.
 * - Space: O(n) to store the scores in an array.
 */
#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;

int main() {
  // n: total number of participants.
  // k: the 1-based rank that determines the cutoff score.
  int n, k;
  cin >> n >> k;

  // Use a C-style Variable-Length Array (VLA) to store scores.
  // Note: VLAs are a non-standard C++ extension, but common in competitive programming.
  int a[n];

  // Read all participant scores into the array.
  // The problem assumes the input scores are in non-increasing order.
  for (int i = 0; i < n; ++i) {
    cin >> a[i];
  }

  // 'ans' will store the final count of advancing participants.
  int ans = 0;

  // The sort is commented out, correctly assuming the input is already sorted.
  // sort(a, a+n);

  /**
   * @block Advancement Counting
   * @logic Iterates through all scores to count how many meet the criteria.
   */
  for (int i = 0; i < n; ++i) {
    // The criteria: score must be positive AND greater than or equal to the
    // score of the k-th person (at index k-1).
    if (a[i] >= a[k - 1] && a[i] > 0)
      ans++;
  }

  // Print the final count.
  cout << ans << "
";
}
