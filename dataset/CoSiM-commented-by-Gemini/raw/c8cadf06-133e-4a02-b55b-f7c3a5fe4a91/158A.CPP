/**
 * @file 158A.CPP
 * @brief Solution for Codeforces problem 158A, "Next Round".
 * @author derekhh
 * @date May 31, 2015
 *
 * @details This program is intended to calculate the number of participants who
 * will advance to the next round of a competition. A participant advances if they
 * achieve a positive score that is greater than or equal to the score of the
 * k-th place finisher.
 *
 * @section Algorithm
 * The code attempts to solve the problem in a single pass, which is logically flawed.
 * It reads the scores of `n` participants and, for each participant `i`, immediately
 * tries to determine if they advance.
 *
 * The condition for advancing is checked as `a[i] > 0 && (i < k || a[i] == a[k - 1])`.
 * The logical error occurs in the `a[i] == a[k - 1]` part. When the loop index `i`
 * is less than `k - 1`, the value of `a[k - 1]` has not yet been read from the input,
 * making the comparison invalid as it uses an uninitialized memory value.
 *
 * A correct approach would be a two-pass algorithm:
 * 1. Read all `n` scores into the array `a`.
 * 2. Get the threshold score from `a[k-1]`.
 * 3. Iterate from `i = 0` to `n-1` and count all `a[i]` such that `a[i] >= a[k-1]` and `a[i] > 0`.
 *
 * @subsection Complexity
 * - Time Complexity: O(n), due to the single for-loop.
 * - Space Complexity: O(n), for storing `n` scores in the array `a` (up to a fixed size of 50).
 */
// 158A.cpp
// A. Next Round
// Author: derekhh
// May 31, 2015

#include <iostream>
using namespace std;

// Global array to store the scores of up to 50 participants.
int a[50];

int main() {
  // n: number of participants, k: the rank that determines the score threshold.
  int n, k;
  cin >> n >> k;
  // ans: counts the number of participants who advance.
  int ans = 0;

  /**
   * @brief This loop reads scores and attempts to count advancing participants in a single pass.
   *
   * @warning Logical Flaw: The condition `a[i] == a[k - 1]` is evaluated before the
   * entire array of scores is populated. For any index `i` less than `k - 1`,
   * `a[k - 1]` contains an indeterminate value, leading to incorrect logic.
   */
  for (int i = 0; i < n; i++) {
    cin >> a[i];
    // The intended logic is to count participants with a positive score who
    // are either in the top k-1 places or have a score equal to the k-th place.
    if (a[i] > 0 && (i < k || a[i] == a[k - 1]))
      ans++;
  }
  cout << ans << endl;
  return 0;
}