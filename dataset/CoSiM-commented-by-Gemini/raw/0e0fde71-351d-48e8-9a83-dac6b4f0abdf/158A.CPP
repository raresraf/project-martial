/**
 * @file 158A.CPP
 * @brief Solves the "Next Round" problem from Codeforces.
 * @details The rule for advancing is that a participant's score must be positive and greater than
 * or equal to the k-th place participant's score. This program determines how many participants advance.
 * It does this in a single pass without storing all the scores, making it highly memory-efficient.
 *
 * @algorithm Single-pass comparison.
 * 1. Read the scores of the first `k` participants. The k-th score is the cutoff score.
 *    If any of these scores are 0, all subsequent scores are also 0, so we can stop early.
 * 2. Read the scores of the remaining `n-k` participants.
 * 3. Count how many of these participants have a score equal to the cutoff score.
 * 4. As soon as a score is less than the cutoff, stop, as all following scores will also be lower.
 * 5. The total number of advancing participants is `k` plus the count from step 3.
 *
 * @complexity
 * Time complexity: O(N), where N is the number of participants, as it reads each score at most once.
 * Space complexity: O(1), as it only stores a few variables.
 */
#include <iostream>

using namespace std;

int main() {
  // n: total number of participants.
  // k: the rank that determines the cutoff score.
  int n, k;
  cin >> n >> k;

  // `a` will store the scores of the first k participants, finally holding the k-th score.
  int a;
  /**
   * @brief First loop: Process the first k participants to find the cutoff score.
   */
  for (int i = 0; i < k; ++i) {
    cin >> a;
    // The "positive score" rule. If a participant at or before the k-th position
    // has a score of 0, no one with a 0 score can advance. Since scores are
    // non-increasing, all subsequent participants also have a score of 0.
    // Thus, only the `i` participants before this one advance.
    if (a == 0) {
      cout << i << endl;
      return 0;
    }
  }

  // `b` will store the scores of participants after the k-th position.
  int b;
  /**
   * @brief Second loop: Process the remaining participants.
   */
  for (int i = k; i < n; ++i) {
    cin >> b;
    // Since scores are non-increasing, the first participant whose score
    // is not equal to the k-th score must have a lower score.
    // This participant and all subsequent ones do not advance.
    // The number of advanced participants is therefore `i`.
    if (b != a) {
      cout << i << endl;
      return 0;
    }
  }

  // If the second loop completes, it means all `n` participants had scores
  // equal to or greater than the k-th participant's positive score.
  // Therefore, all `n` participants advance.
  cout << n << endl;
  return 0;
}
