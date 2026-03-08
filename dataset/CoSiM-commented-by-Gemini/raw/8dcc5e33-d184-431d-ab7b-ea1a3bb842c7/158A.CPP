/**
 * @file 158A.CPP
 * @brief A flawed attempt to solve a contest advancement problem (e.g., Codeforces 158A "Next Round").
 * @details This program tries to count the number of participants who advance, but the logic
 * contains several errors and does not correctly implement the standard problem rules. The goal is
 * to count participants with a positive score that is greater than or equal to the k-th
 * participant's score.
 *
 * @algorithm Single-pass linear scan (with flawed logic).
 * @complexity
 * - Time: O(n), as it iterates through the 'n' scores once.
 * - Space: O(1), as it does not store the full list of scores.
 *
 * @note The logic is incorrect. It mistakenly auto-qualifies the top 'k' participants
 * if their scores are positive. It also fails to handle a zero score at the k-th position
 * and incorrectly checks for equality instead of 'greater than or equal to' for participants
 * ranked lower than k.
 */
#include <iostream>
using namespace std;
int main() {
  // k: the 1-based rank for the cutoff score.
  // n: the total number of participants.
  // num: intended to store the cutoff score from the k-th participant.
  // a: stores the score of the current participant being read.
  // cnt: the counter for advancing participants.
  int k, n, num, a, cnt = 0;
  cin >> n >> k;

  /**
   * @block Main Processing Loop
   * @logic Iterates through each participant's score, attempting to count qualifiers in a single pass.
   */
  for (int i = 1; i <= n; i++) {
    cin >> a;
    
    // BUG: This is incorrect logic. It assumes the first 'k' participants with any positive
    // score automatically advance, which is not the rule.
    if (i <= k && a > 0)
      cnt++;

    // When the k-th participant is reached, their score is stored as the cutoff 'num'.
    // BUG: If the k-th participant's score is 0, 'num' is never initialized, leading to
    // undefined behavior in the next conditional.
    if (i == k) {
      if (a > 0)
        num = a;
    }

    // After passing the k-th participant, check subsequent scores.
    // BUG: This is incorrect. It only counts participants whose score is *exactly equal* to 'num'.
    // The rule requires counting scores that are "greater than or equal to" the cutoff.
    if (i > k && num == a)
      cnt++;
  }
  
  // Output the final, likely incorrect, count.
  cout << cnt;
}
