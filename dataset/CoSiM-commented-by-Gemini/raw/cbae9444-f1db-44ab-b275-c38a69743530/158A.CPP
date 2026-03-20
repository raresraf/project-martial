/**
 * @file 158A.CPP
 * @brief A correct two-pass solution for Codeforces problem 158A, "Next Round".
 * @details This program calculates the number of participants who advance to the
 * next round of a competition. The rule for advancing is that a participant's
 * score must be positive and greater than or equal to the score of the k-th
 * ranked participant.
 *
 * @section Algorithm
 * This solution uses a standard and correct two-pass approach. It also utilizes
 * 1-based indexing for arrays, which is a common preference in competitive programming.
 *
 * 1. Read `n` (the number of participants) and `k` (the rank of the cutoff score).
 * 2. A variable-length array (VLA) `a` of size `n+1` is declared to store scores.
 *    Note that VLAs are a non-standard C++ extension.
 * 3. **First Pass:** A loop reads all `n` scores from the input and populates the
 *    array `a` from index 1 to `n`.
 * 4. **Second Pass:** Another loop iterates through the scores from 1 to `n`. It
 *    compares each participant's score `a[i]` against the cutoff score `a[k]`.
 * 5. A counter `ans` is incremented if a score `a[i]` is both greater than or equal
 *    to `a[k]` and is positive (`a[i] > 0`).
 * 6. The final count `ans` is printed.
 *
 * @subsection Complexity
 * - Time Complexity: O(n), as the code involves two separate loops that each
 *   run `n` times.
 * - Space Complexity: O(n), due to the array used to store the `n` scores.
 */
#include <bits/stdc++.h>
#include <iostream>
using namespace std;
int main() {
  int n, k;
  cin >> n >> k;
  // A variable-length array (VLA) `a` is used to store scores.
  // The size is n+1 to facilitate 1-based indexing (from a[1] to a[n]).
  // Note: VLAs are a non-standard C++ extension.
  int a[n + 1] = {0};
  int ans = 0;

  /**
   * @brief First pass: Read all participant scores into the array.
   * The loop runs from 1 to n, populating the array with 1-based indices.
   */
  for (int i = 1; i <= n; i++) {
    cin >> a[i];
  }

  /**
   * @brief Second pass: Iterate through the scores to count the number of
   * participants who advance.
   * The condition checks if the score is positive and at least as high as the
   * k-th participant's score.
   */
  for (int i = 1; i <= n; i++) {
    if (a[i] >= a[k] && a[i] > 0) {
      ans++;
      // The commented-out line was likely used for debugging purposes.
      // cout<<a[i]<<" ";
    }
  }
  cout << ans;
  return 0;
}