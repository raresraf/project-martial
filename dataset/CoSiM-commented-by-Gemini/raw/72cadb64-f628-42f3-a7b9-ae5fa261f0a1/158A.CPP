/**
 * @file 158A.CPP
 * @brief Solution for Codeforces Problem 158A - "Next Round".
 *
 * This program determines the number of participants who will advance to the
 * next round of a competition. The rule for advancing is that a participant's
 * score must be positive and greater than or equal to the score of the
 * k-th place participant.
 *
 * @algorithm Score filtering and counting.
 * @time_complexity O(N), where N is the number of participants, as the program
 *                  iterates through the scores twice in separate loops.
 * @space_complexity O(N) to store the scores of the participants.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, k, a[105], b, cn = 0;
  // Read the number of participants (n) and the rank (k) for the cutoff score.
  cin >> n >> k;
  /**
   * @brief Reads the scores of all participants.
   * Note: The array is 1-indexed in this implementation.
   */
  for (int i = 1; i <= n; ++i) {
    cin >> a[i];
  }
  // Get the cutoff score from the k-th participant.
  b = a[k];

  /**
   * @brief Counts the number of participants who advance.
   * This loop iterates through all participants' scores. A participant advances
   * if their score is greater than or equal to the cutoff score 'b' and is also
   * greater than 0.
   */
  for (int i = 1; i <= n; ++i) {
    if (a[i] >= b && a[i] > 0)
      ++cn;
  }
  cout << cn << endl;
}