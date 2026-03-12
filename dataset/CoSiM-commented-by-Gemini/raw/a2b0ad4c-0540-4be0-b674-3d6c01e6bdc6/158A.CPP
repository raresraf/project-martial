/**
 * @file 158A.CPP
 * @brief Solution to the "Next Round" problem.
 *
 * This program determines how many participants will advance to the next round of a contest.
 * The rule for advancing is that a participant's score must be positive and greater than
 * or equal to the score of the k-th place finisher. The input scores are guaranteed
 * to be in non-increasing order.
 *
 * @algorithm Linear Scan
 * @time_complexity O(N), where N is the number of participants, due to two separate loops that iterate up to N times.
 * @space_complexity O(N) to store the scores of the participants.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, k, c = 0;
  cin >> n >> k;
  int a[n];
  /**
   * @brief Reads the scores of all participants into an array.
   * The scores are provided in non-increasing order.
   */
  for (int i = 0; i < n; i++) {
    cin >> a[i];
  }

  /**
   * @brief Counts the number of participants who advance to the next round.
   *
   * A participant advances if their score is positive and is greater than or
   * equal to the score of the k-th ranked participant (a[k-1]).
   *
   * Pre-condition: All scores are stored in array `a`.
   * Invariant: `c` holds the count of participants who have qualified so far.
   */
  for (int i = 0; i < n; i++) {
    if (a[i] >= a[k - 1] && a[i] > 0)
      c++;
  }
  cout << c;
  return 0;
}