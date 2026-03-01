/**
 * @file 158A.CPP
 * @brief Solution for the "Next Round" problem.
 *
 * @problem In a contest, 'n' participants are scored. Their scores are given in
 * non-increasing order. A participant advances to the next round if their score is
 * positive and greater than or equal to the score of the k-th place finisher.
 * The task is to count the total number of participants who advance.
 *
 * @algorithm The algorithm leverages the non-increasing order of scores. It determines
 * the score of the k-th participant (`k_score`).
 *
 * Two main cases are considered:
 * 1. If `k_score` is positive:
 *    All of the first 'k' participants have scores at least `k_score` and are positive,
 *    so they all advance. The algorithm then continues to count any participants
 *    after the k-th position who have the exact same score as `k_score`.
 * 2. If `k_score` is 0:
 *    No one from the k-th position onwards can advance, as their scores are also 0 or
 *    less. The algorithm then counts only the participants from the beginning who have
 *    a score greater than 0.
 *
 * @time_complexity O(N) because, in the worst case, it may iterate through all 'n'
 * participants (e.g., for input reading or if all scores are identical).
 * @space_complexity O(N) to store the scores of the 'n' participants.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, k;
  cin >> n >> k;

  // Note: The use of a variable-length array (VLA) `a[n]` is a C99 feature
  // and not standard C++, but is supported as an extension by some compilers.
  int a[n];
  for (int i = 0; i < n; i++)
    cin >> a[i];

  // Get the score of the k-th participant.
  int x = a[k - 1];

  // Case 1: The k-th participant has a positive score.
  if (x > 0) {
    int i = k;
    int cnt = 0;
    // Count how many participants after the k-th have the same score.
    // Because the array is sorted non-increasingly, we only need to check for equality.
    while (a[i] == x && i < n) {
      i++;
      cnt++;
    }
    // The total is the first k participants plus any who tied with the k-th.
    cout << k + cnt << "\n";
  }
  // Case 2: The k-th participant's score is 0.
  else {
    int i = 0;
    int cnt = 0;
    // Count how many participants have a score greater than 0.
    // This loop will stop at the first non-positive score or at the k-th person.
    while (a[i] != 0 && i < k) {
      i++;
      cnt++;
    }
    cout << cnt << "\n";
  }
  return 0;
}
